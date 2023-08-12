import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['str_attention']

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class STRA(nn.Module):
    def __init__(self, inp, oup, groups=32):
        #torch.Size([4, 512, 32, 32])
        #空间方向的Attention 时空坐标注意力
        super(STRA, self).__init__()
        #n.c,h,w h不变 w变成1  w通道注意力
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        #n.c,h,w w不变 h变成1  w通道注意力
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n\n\n\bbbbbbbbbbbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        mip = max(8, inp // groups)
        #(input_size + 2 * padding_size − filter_size)/stride+1
        #只改变通道数量不改变
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.bn2 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        #时空注意力4 1,32,32
        self.conv4 = nn.Conv2d(oup,1,kernel_size=3,stride=1,padding=1)
        self.relu = h_swish()

    def forward(self, x):
        #print(x.shape)
        #torch.Size([4, 512, 32, 32])
        '''
        x_h.shape,x_w.shape: torch.Size([4, 512, 32, 1]) torch.Size([4, 512, 32, 1])
        y: torch.Size([4, 512, 64, 1])
        x_h.shape,x_w.shape2222: torch.Size([4, 16, 32, 1]) torch.Size([4, 16, 32, 1])
        x_h.shape,x_w.shape:3333 torch.Size([4, 16, 32, 1]) torch.Size([4, 16, 1, 32])
        x_h.shape,x_w.shape:444444 torch.Size([4, 512, 32, 32]) torch.Size([4, 512, 32, 32])
        y: torch.Size([4, 512, 32, 32])
        '''
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        #print('x_h.shape,x_w.shape:',x_h.shape,x_w.shape)
        y = torch.cat([x_h, x_w], dim=2)
        #print('y:',y.shape)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        z = self.conv4(identity)
        #print(z.shape)
        z = self.bn2(z)
        z = self.relu(z)
        #print('z:',z.shape)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        #print('x_h.shape,x_w.shape2222:',x_h.shape,x_w.shape)
        x_w = x_w.permute(0, 1, 3, 2)
        #print('x_h.shape,x_w.shape:3333',x_h.shape,x_w.shape)
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        #print('x_h.shape,x_w.shape:444444',x_h.shape,x_w.shape)
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        #print('x_h.shape,x_w.shape:555555',x_h.shape,x_w.shape)
        
        y = identity * x_w * x_h*z
        #y = identity*x_w*x_h
        #print('y:',y.shape)
        return y

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU6(inplace=True),
                # pw-linear
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # coordinate
                STR(hidden_dim, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y

class STR_Attention(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(STR_Attention, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(output_channel, num_classes)
                )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                #print(m.weight.size())
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def str_attention(**kwargs):
    return  STR_Attention(**kwargs)
