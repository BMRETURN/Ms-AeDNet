import torch
import torch.nn as nn


class IEBlock(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_node):
        super(IEBlock, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node

        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )
        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)
        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)

    def forward(self, x):
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class LightTS(nn.Module):
    def __init__(self, configs, chunk_size=24):
        super(LightTS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # Chunk processing setup
        self.chunk_size = min(configs.pred_len, configs.seq_len, chunk_size)
        if self.seq_len % self.chunk_size != 0:  # Pad sequence if needed
            self.seq_len += (self.chunk_size - self.seq_len % self.chunk_size)
        self.num_chunks = self.seq_len // self.chunk_size

        # Architecture parameters
        self.d_model = configs.hidden_dim
        self._build()

    def _build(self):
        # Feature extraction modules
        self.layer_1 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )
        self.layer_2 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )

        # Projection layers
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        # Final prediction module
        self.layer_3 = IEBlock(
            input_dim=self.d_model // 2,
            hid_dim=self.d_model // 2,
            output_dim=self.pred_len,
            num_node=self.enc_in
        )

        # Auxiliary linear layer
        self.ar = nn.Linear(self.seq_len, self.pred_len)

    def encoder(self, x):
        B, T, N = x.shape

        # Highway connection
        highway = self.ar(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Continuous sampling branch
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1).reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(-1)

        # Interval sampling branch
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2).reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(-1)

        # Feature fusion
        x3 = torch.cat([x1, x2], dim=-1).reshape(B, N, -1).permute(0, 2, 1)
        out = self.layer_3(x3)

        return out + highway

    def forward(self, x_enc):
        # Normalization
        B, L, _ = x_enc.shape
        x = x_enc[:, :, :self.enc_in]
        # means = x .mean(1, keepdim=True).detach()
        # stdev = torch.sqrt(torch.var(x , dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        # x = (x  - means) / stdev

        # Feature extraction
        dec_out = self.encoder(x)

        # Denormalization
        return dec_out #* stdev + means