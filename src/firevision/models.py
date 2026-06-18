from __future__ import annotations


def build_classifier(name: str = "resnet18", num_classes: int = 2, pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models

    if name == "resnet18":
        model = models.resnet18(weights="DEFAULT" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported classifier: {name}")
    return model


def build_temporal_model(
    name: str = "cnn_lstm",
    num_classes: int = 2,
    hidden_size: int = 256,
    pretrained: bool = True,
):
    import torch.nn as nn
    from torchvision import models

    if name == "video_swin_t":
        model = models.video.swin3d_t(weights="DEFAULT" if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        model.expects_bcthw = True
        return model
    if name != "cnn_lstm":
        raise ValueError(f"Unsupported temporal model: {name}")

    class CNNLSTM(nn.Module):
        expects_bcthw = False

        def __init__(self):
            super().__init__()
            backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
            feature_size = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.lstm = nn.LSTM(feature_size, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            batch, time, channels, height, width = x.shape
            features = self.backbone(x.reshape(batch * time, channels, height, width))
            features = features.reshape(batch, time, -1)
            output, _ = self.lstm(features)
            return self.head(output[:, -1])

    return CNNLSTM()
