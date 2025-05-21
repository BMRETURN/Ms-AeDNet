import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_dim = configs.patch_dim
        self.patch_head = configs.heads
        self.enc_in = configs.enc_in

        self.patch_encoder = nn.Linear(self.patch_size, self.patch_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.patch_dim,
            num_heads=self.patch_head,
            batch_first=True
        )
        self.predictor = nn.Linear(self.patch_dim*self.patch_num, self.pred_len)


    def forward(self, x):
        B, L, _ = x.shape
        x = x.permute(0, 2, 1)  # (B, M, L)

        # Normalization
        voltage_data = x[:, :self.enc_in, :]  # (B,5,L)
        means = voltage_data.mean(-1, keepdim=True).detach()  # (B,5,L)
        stdev = torch.sqrt(torch.var(voltage_data, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized_voltage = (voltage_data - means) / stdev

        # Patchify
        x = self.padding_patch_layer(normalized_voltage)  # (B, M, L + stride)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # (B, M, num_patches, patch_size)
        emd_x = self.patch_encoder(x).reshape(B*self.enc_in, self.patch_num, self.patch_dim)

        attn_output, _ = self.attention(emd_x, emd_x, emd_x)
        attn_output = attn_output.reshape(B, self.enc_in, self.patch_dim*self.patch_num)

        # Prediction
        prediction = self.predictor(attn_output)
        predictions = prediction * stdev + means

        return predictions.permute(0, 2, 1)