import torch
import torch.nn as nn

class MHSA(nn.Module):
    """ 
    input:
        feat:B, G, T, C
      
    """
    def __init__(self, num_heads, input_dim, ratio=2, thr=0.6):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.out_dim = self.input_dim * ratio
        self.head_dim = input_dim // num_heads
        self.thr = thr
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        # Define linear layers for query, key, value, and output projections
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, input_dim)
        self.output_projection = nn.Linear(input_dim,  self.out_dim)
        self.offset_projection = nn.Conv1d(self.out_dim, 3+self.out_dim, 1)
        self.bn = nn.BatchNorm1d(self.out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def split_heads(self, x, B, G):
        # Reshape input tensor to (batch_size, num_heads, seq_len, head_dim)
        x = x.view(B*G, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x, y):
        B, G, T, C= x.size() # B, G, T, C
        
        # Split input tensor into heads
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        
        query = self.split_heads(query, B, G) # B*G, H, T, C/H
        key = self.split_heads(key, B, G)
        value = self.split_heads(value, B, G)
        
        # Calculate self-attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5) # B*G, H, T, T
        # Calculate attention weights
        attention_weights = scores.softmax(dim=-1) # B*G, H, T, T
        
        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, value)
        
        # Concatenate and project output for each head
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(B*G, T, C)
        value = self.out_projection(attended_values).view(B, G, T, -1)  # B, G, T, C'

        scores = torch.matmul(scores, scores.transpose(-2, -1))
        upper = torch.triu(torch.ones_like(scores), diagonal=1) # scores
        upper = torch.mean(upper, dim=1).squeeze().softmax(dim=-1) # B, G, T, T

        mask = upper > self.thr

        for b in range(B):
            for g in range(G):
                for t in range(T):
                    ind = torch.nonzero(mask[b, g, t])
                    feat = value[b,g,ind].t() # C', T'
                    offset = self.offset_projection(feat)
                    feat_offset = self.bn(offset[3:])
                    feat_offset = self.pool(feat + feat_offset) # C', 1
                    coor_offset = offset[0:3]
                    coor_offset = self.pool(y+coor_offset).squeeze() # 3, 1
                    coor_offset = coor_offset/self.grid_size
                    coor_offset = self.pool(coor_offset).int().tolist()
                

        return output

# Example usage
B, N, C = 2, 8, 16  # Batch size, sequence length, input dimension
H = 4  # Number of heads

# Create a random input tensor
input_tensor = torch.randn(B, N, C)

# Create the multi-head attention layer
multihead_attention = MHSA(num_heads=H, input_dim=C)

# Apply the multi-head attention layer to the input tensor
output = multihead_attention(input_tensor)

print(output.shape)  # Output shape will be (B, N, C)
