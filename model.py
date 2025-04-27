import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net

    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out

# Modified ResNet1D to output features before final pooling and dense layer
class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super(ResNetBackbone, self).__init__()
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap

        # First block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # Residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            is_first_block = (i_block == 0)
            downsample = (i_block % self.downsample_gap == 1)
            
            if is_first_block:
                in_channels_block = base_filters
                out_channels_block = in_channels_block
            else:
                in_channels_block = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels_block = in_channels_block * 2
                else:
                    out_channels_block = in_channels_block

            tmp_block = BasicBlock(
                in_channels=in_channels_block,
                out_channels=out_channels_block,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block
            )
            self.basicblock_list.append(tmp_block)

        # Final layers
        self.final_bn = nn.BatchNorm1d(out_channels_block)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        if self.verbose:
            print('Input shape:', out.shape)
        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        for i_block in range(self.n_block):
            out = self.basicblock_list[i_block](out)
            if self.verbose:
                print(f'Block {i_block} output shape:', out.shape)

        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        return out

# EEGTransformer with corrected FFN output dimension
class EEGTransformer(nn.Module):
    def __init__(self, num_channels, num_timepoints, output_dim, hidden_dim, num_heads, key_query_dim, intermediate_dim):
        super(EEGTransformer, self).__init__()
        
        # Positional Encoding
        self.positional_encoding = torch.zeros(num_channels, num_timepoints)
        for j in range(num_channels):
            for k in range(num_timepoints):
                if j % 2 == 0:
                    self.positional_encoding[j][k] = torch.sin(torch.tensor(k) / (10000 ** (j / num_channels)))
                else:
                    self.positional_encoding[j][k] = torch.cos(torch.tensor(k) / (10000 ** ((j - 1) / num_channels)))
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=num_channels,
            num_heads=num_heads
        )
        
        # FFN to ensure output dimension matches input
        self.ffn = nn.Sequential(
            nn.Linear(num_channels, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, num_channels)  # Ensure output matches num_channels
        )
        
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        
        self.classifier = nn.Linear(num_channels * num_timepoints, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # Input standardization
        mean = X.mean(dim=2, keepdim=True)
        std = X.std(dim=2, keepdim=True)
        X_hat = (X - mean) / (std + 1e-5)
        
        # Add positional encoding
        X_tilde = X_hat + self.positional_encoding.to(X.device)
        
        # Permute for multihead attention: (seq_len, batch_size, embed_dim)
        X_tilde = X_tilde.permute(2, 0, 1)
        
        # Multi-head self-attention
        attn_output, _ = self.multihead_attn(X_tilde, X_tilde, X_tilde)
        
        # Apply layer norm
        attn_output = self.norm1(attn_output)
        
        # FFN
        ff_output = self.ffn(attn_output)
        ff_output = self.norm2(ff_output + attn_output)  # Residual connection
        
        # Reshape back: (batch_size, num_channels, num_timepoints)
        ff_output = ff_output.permute(1, 2, 0)
        
        # Flatten and classify
        ff_output_flat = ff_output.contiguous().view(ff_output.size(0), -1)
        output = self.classifier(ff_output_flat)
        return output.squeeze(1)

        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net


# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, resnet_config, transformer_config):
        super(CombinedModel, self).__init__()
        self.resnet = ResNetBackbone(**resnet_config)
        
        # Determine ResNet output dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, resnet_config['in_channels'], resnet_config.get('input_length', 2000))
            dummy_out = self.resnet(dummy_input)
            C, T = dummy_out.shape[1], dummy_out.shape[2]
        
        # Update transformer_config with correct dimensions
        transformer_config.update({
            'num_channels': C,
            'num_timepoints': T,
            'ffn_output_dim': C  # Ensure FFN outputs correct dimension
        })
        # Adjust transformer_config to match EEGTransformer parameters
        self.transformer = EEGTransformer(
            num_channels=C,
            num_timepoints=T,
            output_dim=transformer_config['output_dim'],
            hidden_dim=transformer_config.get('hidden_dim', 64),
            num_heads=transformer_config['num_heads'],
            key_query_dim=transformer_config.get('key_query_dim', 64),
            intermediate_dim=transformer_config['intermediate_dim']
        )
    
    def forward(self, x):
        x = x.squeeze(1)
        # x = x.permute(0, 2, 1)
        features = self.resnet(x)
        output = self.transformer(features)
        return output
    

resnet_config = {
    'in_channels': 19,
    'base_filters': 64,
    'kernel_size': 17,
    'stride': 2,
    'groups': 1,
    'n_block': 4,
    'downsample_gap': 2,
    'increasefilter_gap': 4,
    'use_bn': True,
    'use_do': True,
    # 'input_length': 2000,
    'verbose': False
}

transformer_config = {
    'output_dim': 1,
    'hidden_dim': 32,
    'num_heads': 4,
    'key_query_dim': 32,
    'intermediate_dim': 128
}