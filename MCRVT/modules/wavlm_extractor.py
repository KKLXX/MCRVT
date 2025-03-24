import torch
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2Processor  # 使用HuggingFace官方实现


class WavLMFeatureExtractor(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base"):
        super().__init__()
        # 加载预训练模型和处理器
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.wavlm = WavLMModel.from_pretrained(model_name)

        # 冻结模型参数（可选）
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(self, raw_audio):
        """
        输入: raw_audio - 原始音频波形 [batch, time_samples]
        输出: 特征序列 [batch, time_frames, feature_dim]
        """
        # 1. 标准化音频输入
        inputs = self.processor(
            raw_audio,
            sampling_rate=16000,  # 假设16kHz，根据实际数据调整
            return_tensors="pt",
            padding=True
        ).input_values.to(raw_audio.device)

        # 2. 提取特征
        outputs = self.wavlm(inputs)
        return outputs.last_hidden_state  # [batch, time_frames, 768]