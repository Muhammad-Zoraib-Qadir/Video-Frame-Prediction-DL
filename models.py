import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_msssim import ssim
import torch.nn.functional as F
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
import torch.nn.functional as F
from pytorch_msssim import ssim
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_msssim import ssim
class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size=3, stride=1, layer_norm=False):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = 1
        self._forget_bias = 1.0
        
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
            self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
            self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
            self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
            
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

class PredRNN(pl.LightningModule):
    def __init__(self, input_channels=1, num_hidden=[64, 64, 64, 64], input_frames=10, output_frames=5):
        super(PredRNN, self).__init__()
        
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.num_layers = len(num_hidden)
        self.num_hidden = num_hidden
        self.frame_channel = input_channels
        self.width = 64  
        
        # Initialize LSTM cells
        cell_list = []
        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.width)
            )
        self.cell_list = nn.ModuleList(cell_list)
        
        # Output projection
        self.conv_last = nn.Conv2d(num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Loss criteria
        self.mse_criterion = nn.MSELoss()
        
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, frames):
        batch_size = frames.size(0)
        
        # Initialize states
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch_size, self.num_hidden[i], self.width, self.width]).to(frames.device)
            h_t.append(zeros)
            c_t.append(zeros)
        
        memory = torch.zeros([batch_size, self.num_hidden[0], self.width, self.width]).to(frames.device)
        
        # Encoding
        for t in range(self.input_frames):
            x_t = frames[:, t]
            h_t[0], c_t[0], memory = self.cell_list[0](x_t, h_t[0], c_t[0], memory)
            
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], memory)
        
        # Prediction
        next_frames = []
        for t in range(self.output_frames):
            x_t = self.conv_last(h_t[-1])
            h_t[0], c_t[0], memory = self.cell_list[0](x_t, h_t[0], c_t[0], memory)
            
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], memory)
                
            next_frames.append(self.conv_last(h_t[-1]))
        
        # Stack predictions
        predictions = torch.stack(next_frames, dim=1)
        return torch.sigmoid(predictions)

    def calculate_metrics(self, pred, target):
        """Calculate MSE and SSIM metrics"""
        mse = self.mse_criterion(pred, target)
        ssim_val = ssim(pred, target, data_range=1.0)
        return mse, ssim_val

    def training_step(self, batch, batch_idx):
        input_seq = batch['input']
        target_seq = batch['target']
        
        predictions = self(input_seq)
        mse_loss, ssim_val = self.calculate_metrics(predictions, target_seq)
        
        # Log metrics
        self.log('train_mse', mse_loss, prog_bar=True)
        self.log('train_ssim', ssim_val, prog_bar=True)
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        input_seq = batch['input']
        target_seq = batch['target']
        
        predictions = self(input_seq)
        mse_loss, ssim_val = self.calculate_metrics(predictions, target_seq)
        
        self.log('val_mse', mse_loss, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        
        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_mse"
        }
    



class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.reset_metrics()

    def reset_metrics(self):
        self.epoch_mse = []
        self.epoch_ssim = []
        self.epoch_combined_loss = []
        self.val_epoch_mse = []
        self.val_epoch_ssim = []
        self.val_epoch_combined_loss = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.reset_metrics()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.epoch_mse.append(outputs['mse'].item())
        self.epoch_ssim.append(outputs['ssim'].item())
        self.epoch_combined_loss.append(outputs['combined_loss'].item())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_epoch_mse.append(outputs['val_mse'].item())
        self.val_epoch_ssim.append(outputs['val_ssim'].item())
        self.val_epoch_combined_loss.append(outputs['val_combined_loss'].item())

    def on_train_epoch_end(self, trainer, pl_module):
        avg_mse = sum(self.epoch_mse) / len(self.epoch_mse)
        avg_ssim = sum(self.epoch_ssim) / len(self.epoch_ssim)
        avg_combined_loss = sum(self.epoch_combined_loss) / len(self.epoch_combined_loss)
        
        avg_val_mse = sum(self.val_epoch_mse) / len(self.val_epoch_mse) if self.val_epoch_mse else 0
        avg_val_ssim = sum(self.val_epoch_ssim) / len(self.val_epoch_ssim) if self.val_epoch_ssim else 0
        avg_val_combined_loss = sum(self.val_epoch_combined_loss) / len(self.val_epoch_combined_loss) if self.val_epoch_combined_loss else 0

        print("\n" + "="*50)
        print(f"Epoch {trainer.current_epoch+1} Metrics:")
        print("-"*20 + " Training " + "-"*20)
        print(f"MSE Loss:       {avg_mse:.4f}")
        print(f"SSIM Score:     {avg_ssim:.4f}")
        print(f"Combined Loss:  {avg_combined_loss:.4f}")
        
        print("-"*20 + "Validation" + "-"*20)
        print(f"MSE Loss:       {avg_val_mse:.4f}")
        print(f"SSIM Score:     {avg_val_ssim:.4f}")
        print(f"Combined Loss:  {avg_val_combined_loss:.4f}")
        print("="*50 + "\n")

        # Log to tensorboard
        pl_module.log('epoch_avg_mse', avg_mse)
        pl_module.log('epoch_avg_ssim', avg_ssim)
        pl_module.log('epoch_avg_combined_loss', avg_combined_loss)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                             out_channels=4 * self.hidden_dim,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

def calculate_metrics(pred, target):
    # Calculate MSE
    mse = F.mse_loss(pred, target)
    
    # Calculate SSIM
    # SSIM expects inputs in range [0, 1]
    ssim_value = ssim(pred, target, data_range=1.0, size_average=True)
    
    return mse, ssim_value

class VideoPredictor(pl.LightningModule):
    def __init__(self, nf=64, input_frames=10, output_frames=5, input_channels=1):
        super(VideoPredictor, self).__init__()
        
        self.input_frames = input_frames
        self.output_frames = output_frames
        print(nf,input_channels)
        self.encoder_1_convlstm = ConvLSTMCell(
            input_dim=input_channels,  # 1 for grayscale, 3 for RGB
            hidden_dim=nf,
            kernel_size=(3, 3),
            bias=True
        )
        
        self.encoder_2_convlstm = ConvLSTMCell(
            input_dim=nf,
            hidden_dim=nf,
            kernel_size=(3, 3),
            bias=True
        )
        
        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=nf,
            hidden_dim=nf,
            kernel_size=(3, 3),
            bias=True
        )
        
        self.decoder_2_convlstm = ConvLSTMCell(
            input_dim=nf,
            hidden_dim=nf,
            kernel_size=(3, 3),
            bias=True
        )
        
        self.decoder_CNN = nn.Conv3d(
            in_channels=nf,
            out_channels=input_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

    def forward(self, x):
        b, seq_len, c, h, w = x.size()
        
        # Initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # Encoding
        for t in range(seq_len):
            current_input = x[:, t]
            h_t, c_t = self.encoder_1_convlstm(current_input, [h_t, c_t])
            h_t2, c_t2 = self.encoder_2_convlstm(h_t, [h_t2, c_t2])

        # Decoder
        outputs = []
        encoder_vector = h_t2
        
        for t in range(self.output_frames):
            h_t3, c_t3 = self.decoder_1_convlstm(encoder_vector, [h_t3, c_t3])
            h_t4, c_t4 = self.decoder_2_convlstm(h_t3, [h_t4, c_t4])
            encoder_vector = h_t4
            outputs.append(h_t4)
        
        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.sigmoid(outputs)
        
        final_output = outputs.permute(0, 2, 1, 3, 4)
        return final_output

    def training_step(self, batch, batch_idx):
        input_seq = batch['input']
        target_seq = batch['target']
        
        predictions = self(input_seq)
        
        # Calculate both metrics
        mse_loss, ssim_value = calculate_metrics(predictions, target_seq)
        
        # Combined loss
        combined_loss = mse_loss - 0.5 * ssim_value
        
        # Log batch metrics
        self.log('train_mse', mse_loss)
        self.log('train_ssim', ssim_value)
        self.log('train_combined_loss', combined_loss)
        
        # Return dictionary with all metrics
        return {
            'loss': combined_loss,
            'mse': mse_loss,
            'ssim': ssim_value,
            'combined_loss': combined_loss
        }

    def validation_step(self, batch, batch_idx):
        input_seq = batch['input']
        target_seq = batch['target']
        
        predictions = self(input_seq)
        
        # Calculate both metrics
        mse_loss, ssim_value = calculate_metrics(predictions, target_seq)
        
        # Combined loss
        combined_loss = mse_loss - 0.5 * ssim_value
        
        # Log metrics
        self.log('val_mse', mse_loss)
        self.log('val_ssim', ssim_value)
        self.log('val_combined_loss', combined_loss)
        
        return {
            'val_loss': combined_loss,
            'val_mse': mse_loss,
            'val_ssim': ssim_value,
            'val_combined_loss': combined_loss
        }

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)
    




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VideoTransformer(pl.LightningModule):
    def __init__(self, input_channels=1, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, input_frames=10, output_frames=5):
        super().__init__()
        
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.d_model = d_model
        
        # CNN Encoder for each frame
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Frame Decoder (CNN)
        self.frame_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.save_hyperparameters()

    def encode_frames(self, x):
        # x shape: [batch, frames, channels, height, width]
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Encode each frame
        encoded_frames = []
        for i in range(seq_len):
            encoded = self.frame_encoder(x[:, i])  # [batch, d_model, h', w']
            h, w = encoded.shape[-2:]
            encoded = encoded.flatten(2).transpose(1, 2)  # [batch, h'*w', d_model]
            encoded_frames.append(encoded)
        
        encoded_sequence = torch.stack(encoded_frames, dim=1)  # [batch, seq_len, h'*w', d_model]
        return encoded_sequence, (h, w)

    def forward(self, x):
        # Encode input frames
        encoded_sequence, (h, w) = self.encode_frames(x)  # [batch, seq_len, h'*w', d_model]
        batch_size = encoded_sequence.shape[0]
        
        # Reshape for transformer
        encoded_sequence = encoded_sequence.flatten(1, 2)  # [batch, seq_len*h'*w', d_model]
        
        # Add positional encoding
        encoded_sequence = self.pos_encoder(encoded_sequence)
        
        # Transformer encoder
        memory = self.transformer_encoder(encoded_sequence)
        
        # Generate decoder input (start tokens)
        decoder_input = torch.zeros(
            batch_size,
            self.output_frames * h * w,
            self.d_model,
            device=x.device
        )
        decoder_input = self.pos_encoder(decoder_input)
        
        # Transformer decoder
        output = self.transformer_decoder(decoder_input, memory)
        
        # Reshape output and decode frames
        output = output.view(batch_size, self.output_frames, h * w, self.d_model)
        
        predicted_frames = []
        for i in range(self.output_frames):
            current = output[:, i].view(batch_size, h, w, self.d_model)
            current = current.permute(0, 3, 1, 2)
            decoded = self.frame_decoder(current)
            predicted_frames.append(decoded)
        
        # Stack predictions
        predictions = torch.stack(predicted_frames, dim=1)
        return predictions

    def calculate_metrics(self, pred, target):
        """Calculate MSE and SSIM metrics"""
        mse = F.mse_loss(pred, target)
        ssim_val = ssim(pred, target, data_range=1.0)
        return mse, ssim_val

    def training_step(self, batch, batch_idx):
        input_seq = batch['input']
        target_seq = batch['target']
        
        predictions = self(input_seq)
        mse_loss, ssim_val = self.calculate_metrics(predictions, target_seq)
        
        self.log('train_mse', mse_loss, prog_bar=True)
        self.log('train_ssim', ssim_val, prog_bar=True)
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        input_seq = batch['input']
        target_seq = batch['target']
        
        predictions = self(input_seq)
        mse_loss, ssim_val = self.calculate_metrics(predictions, target_seq)
        
        self.log('val_mse', mse_loss, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        
        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_mse"
        }

