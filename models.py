import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from config import Config

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x).view(B, -1, W*H).permute(0,2,1)
        key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(B, -1, W*H)

        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out
    
class Generator(nn.Module):
    def __init__(self, latent_dim, use_spectral_norm=False, use_self_attention=False):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.use_self_attention = use_self_attention

        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 512*4*4),
            nn.BatchNorm1d(512*4*4),
            nn.ReLU(inplace=True),
        )

        def conv_block(in_ch, out_ch, use_spectral_norm=False):
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if use_spectral_norm:
                layers[0] = spectral_norm(layers[0])
            return nn.Sequential(*layers)
        self.conv1 = conv_block(512,256, use_spectral_norm)
        self.conv2 = conv_block(256, 128, use_spectral_norm)

        if use_self_attention:
            self.attention = SelfAttention(128)
        
        self.conv3 = conv_block(128, 64, use_spectral_norm)
        self.conv4 = conv_block(64, 32, use_spectral_norm)    
        self.conv5 = conv_block(32, 16, use_spectral_norm) 

        final_layer = nn.ConvTranspose2d(16,3,4,2,1, bias=False)
        if use_spectral_norm:
            final_layer = spectral_norm(final_layer)

        self.final = nn.Sequential(final_layer, nn.Tanh())

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        x = self.initial(z)
        x = x.view(x.size(0), 512, 4, 4)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.use_self_attention:
            x = self.attention(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final(x)
        
        return x
    
class Discriminator(nn.Module):
    def __init__(self, use_spectral_norm=True, use_self_attention=False):
        super(Discriminator, self).__init__()
        self.use_self_attention = use_self_attention

        def conv_block(in_ch, out_ch, use_spectral_norm=True, use_bn = False):
            layers = [
                nn.Conv2d(in_ch, out_ch, 4,2,1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if use_bn:
                layers.insert(1, nn.BatchNorm2d(out_ch))
            if use_spectral_norm:
                layers[0] = spectral_norm(layers[0])
            return nn.Sequential(*layers)

        self.conv1 = conv_block(3, 64, use_spectral_norm, False)    
        self.conv2 = conv_block(64, 128, use_spectral_norm, False)  
        self.conv3 = conv_block(128, 256, use_spectral_norm, False)

        if use_self_attention:
            self.attention = SelfAttention(256)
        
        self.conv4 = conv_block(256, 512, use_spectral_norm, False) 
        self.conv5 = conv_block(512, 1024, use_spectral_norm, False)

        final_layer = nn.Conv2d(1024, 1,4,2,1, bias=False)
        if use_spectral_norm:
            final_layer = spectral_norm(final_layer)
        
        self.final = final_layer

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        if self.use_self_attention:
            x = self.attention(x)
            
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final(x)
        
        return x.view(x.size(0), -1)
    
def create_models(config):
    generator = Generator(
        latent_dim=config.LATENT_DIM,
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        use_self_attention=config.USE_SELF_ATTENTION
    ).to(config.DEVICE)

    discriminator = Discriminator(
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        use_self_attention=config.USE_SELF_ATTENTION
    ).to(config.DEVICE)

    return generator, discriminator