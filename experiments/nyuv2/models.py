from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from methods import Causals_Attention_patchs

def split_and_flatten_patches(image_tensor, num_patches=64):
    """
    分割图像为多个patches，并将每个patch拉平成一维的数据。
    
    参数:
    image_tensor: 形状为 [batch_size, channels, height, width] 的图像张量
    num_patches: 想要分割成的patch数量，默认值为64
    
    返回:
    拉平后的patches张量，形状为 [batch_size * num_patches, channels * patch_height * patch_width]
    """
    # 确保输入是一个四维张量 [batch_size, channels, height, width]
    if len(image_tensor.shape) != 4:
        raise ValueError("Input tensor must be a 4D tensor.")
    
    batch_size, channels, height, width = image_tensor.shape
    
    # 计算每行和每列的patch数量
    patches_per_side = int(num_patches ** 0.5)
    if patches_per_side * patches_per_side != num_patches:
        raise ValueError("The number of patches must be a perfect square.")
        
    patch_height = height // patches_per_side
    patch_width = width // patches_per_side
    
    if height % patches_per_side != 0 or width % patches_per_side != 0:
        raise ValueError("The image dimensions are not divisible by the number of patches per side.")
    
    # 使用unfold方法来创建patches
    patches = image_tensor.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_height, patch_width)
    patches = patches.permute(0, 2, 1, 3, 4)  # 调整维度顺序以适应后续处理
    
    # 将每个batch中的patches拉平为一维向量
    flattened_patches = patches.reshape(batch_size, num_patches, -1)
    
    # 如果需要，我们可以返回一个二维张量，其中第一维是批次数乘以patch数
    #flattened_patches = flattened_patches.view(-1, flattened_patches.size(-1))
    
    return flattened_patches


def restore_and_merge_patches(flattened_patches, original_shape):

    batch_size, channels, height, width = original_shape
    #print(batch_size)
    # 计算每行和每列的patch数量
    patches_per_side = int((flattened_patches.shape[1]) ** 0.5)
    if patches_per_side * patches_per_side != flattened_patches.shape[1]:
        raise ValueError("The number of patches per image must be a perfect square.")
        
    patch_height = height // patches_per_side
    patch_width = width // patches_per_side
    
    # 检查是否可以完美重组
    if flattened_patches.shape[1] != patches_per_side ** 2:
        raise ValueError("The number of patches does not match the expected amount for perfect reassembly.")
    
    # 确保输入张量是连续的
    if not flattened_patches.is_contiguous():
        flattened_patches = flattened_patches.contiguous()
    
    # 使用.reshape()方法将每个patch恢复为原来的三维形状
    restored_patches = flattened_patches.view(batch_size, patches_per_side * patches_per_side, channels, patch_height, patch_width)
    
    # 重塑和转置以匹配原始图像的布局
    merged = restored_patches.view(batch_size, patches_per_side, patches_per_side, channels, patch_height, patch_width)
    merged = merged.permute(0, 3, 1, 4, 2, 5).contiguous()  # 调整维度顺序以便于重组
    merged = merged.view(original_shape)  # 重塑为原始图像尺寸
    
    return merged



class _SegNet(nn.Module):
    """SegNet MTAN"""

    def __init__(self,loc):
        super(_SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.loc=loc
        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(
                    nn.Sequential(
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                    )
                )
                self.conv_block_dec.append(
                    nn.Sequential(
                        self.conv_layer([filter[i], filter[i]]),
                        self.conv_layer([filter[i], filter[i]]),
                    )
                )

        # define task attention layers
        self.encoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])]
        )
        self.decoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])]
        )
        self.encoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[1]])]
        )
        self.decoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[0]])]
        )

        for j in range(3):
            if j < 2:
                self.encoder_att.append(
                    nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])
                )
                self.decoder_att.append(
                    nn.ModuleList(
                        [self.att_layer([2 * filter[0], filter[0], filter[0]])]
                    )
                )
            for i in range(4):
                self.encoder_att[j].append(
                    self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]])
                )
                self.decoder_att[j].append(
                    self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]])
                )

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 2]])
                )
                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i]])
                )
            else:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        if self.loc==True:
            self.up1=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1,stride=1,padding=0)
            self.up2=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1,stride=1,padding=0)
            self.up3=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1,stride=1,padding=0)
            self.cal=Causals_Attention_patchs(3, 5184, 64)




    def shared_modules(self):
        return [
            self.encoder_block,
            self.decoder_block,
            self.conv_block_enc,
            self.conv_block_dec,
            # self.encoder_att, self.decoder_att,
            self.encoder_block_att,
            self.decoder_block_att,
            self.down_sampling,
            self.up_sampling,
        ]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[0],
                    kernel_size=3,
                    padding=1,
                ),
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=1,
                    padding=0,
                ),
            )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channel[0],
                out_channels=channel[1],
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channel[1],
                out_channels=channel[2],
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = (
            [0] * 5 for _ in range(5)
        )
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1]
                    )
                    atten_encoder[i][j][2] = F.max_pool2d(
                        atten_encoder[i][j][2], kernel_size=2, stride=2
                    )
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1)
                    )
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1]
                    )
                    atten_encoder[i][j][2] = F.max_pool2d(
                        atten_encoder[i][j][2], kernel_size=2, stride=2
                    )

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_encoder[i][-1][-1],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0]
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1)
                    )
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_decoder[i][j - 1][2],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0]
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1)
                    )
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        if self.loc==True:
            # print('loc')
            num_patches=64
          
            patches=split_and_flatten_patches(x,num_patches)   # 16, 64 patch,256 patch 
           
        
            out=self.cal(patches) # patches: bs,64,5184  out[9] [bs,64,5184]
            #self.TEM=self.cal.tem
            restored_x1 = restore_and_merge_patches(out[0], x.shape)
            restored_x2 = restore_and_merge_patches(out[1], x.shape)
            restored_x3 = restore_and_merge_patches(out[2], x.shape)
            
     
            out1=self.up1(restored_x1)      
            out2=self.up2(restored_x2)
            out3=self.up3(restored_x3)
          
            atten_decoder[0][-1][-1] = atten_decoder[0][-1][-1] + out1
            atten_decoder[1][-1][-1] = atten_decoder[1][-1][-1] + out2
            atten_decoder[2][-1][-1] = atten_decoder[2][-1][-1] + out3



        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return (
            [t1_pred, t2_pred, t3_pred],
            (
                atten_decoder[0][-1][-1],
                atten_decoder[1][-1][-1],
                atten_decoder[2][-1][-1],
            ),
        )


class SegNetMtan(nn.Module):
    def __init__(self,loc):
        super().__init__()
        self.segnet = _SegNet(loc)

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" not in n)

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" in n)

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        return []

    def forward(self, x, return_representation=False):
        if return_representation:
            return self.segnet(x)
        else:
            pred, rep = self.segnet(x)
            return pred


class SegNetSplit(nn.Module):
    def __init__(self, model_type="standard"):
        super(SegNetSplit, self).__init__()
        # initialise network parameters
        assert model_type in ["standard", "wide", "deep"]
        self.model_type = model_type
        if self.model_type == "wide":
            filter = [64, 128, 256, 512, 1024]
        else:
            filter = [64, 128, 256, 512, 512]

        self.class_nb = 13

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(
                    nn.Sequential(
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                    )
                )
                self.conv_block_dec.append(
                    nn.Sequential(
                        self.conv_layer([filter[i], filter[i]]),
                        self.conv_layer([filter[i], filter[i]]),
                    )
                )

        # define task specific layers
        self.pred_task1 = nn.Sequential(
            nn.Conv2d(
                in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1
            ),
            nn.Conv2d(
                in_channels=filter[0],
                out_channels=self.class_nb,
                kernel_size=1,
                padding=0,
            ),
        )
        self.pred_task2 = nn.Sequential(
            nn.Conv2d(
                in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1
            ),
            nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0),
        )
        self.pred_task3 = nn.Sequential(
            nn.Conv2d(
                in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1
            ),
            nn.Conv2d(in_channels=filter[0], out_channels=3, kernel_size=1, padding=0),
        )

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # define convolutional block
    def conv_layer(self, channel):
        if self.model_type == "deep":
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=channel[1],
                    out_channels=channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        return conv_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = (
            [0] * 5 for _ in range(5)
        )
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # global shared encoder-decoder network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(g_decoder[i][1]), dim=1)
        t2_pred = self.pred_task2(g_decoder[i][1])
        t3_pred = self.pred_task3(g_decoder[i][1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], g_decoder[i][
            1
        ]  # NOTE: last element is representation


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.segnet = SegNetSplit()

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" not in n)

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" in n)

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        return self.segnet.conv_block_dec[-5].parameters()

    def forward(self, x, return_representation=False):
        if return_representation:
            return self.segnet(x)
        else:
            pred, rep = self.segnet(x)
            return pred
