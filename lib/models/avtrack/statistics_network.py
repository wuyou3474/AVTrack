import torch.nn as nn
import torch



def tile_and_concat(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Merge 1D and 2D tensor (use to aggregate feature maps and representation
    and compute local mutual information estimation)

    Args:
        tensor (torch.Tensor): 2D tensor (feature maps)
        vector (torch.Tensor): 1D tensor representation

    Returns:
        torch.Tensor: Merged tensor (2D)
    """

    B, C, H, W = tensor.size()
    vector = vector.unsqueeze(2).unsqueeze(2)   # [64, 64]->[64, 64, 1, 1]
    expanded_vector = vector.expand((B, vector.size(1), H, W))
    return torch.cat([tensor, expanded_vector], dim=1)


class LocalStatisticsNetwork(nn.Module):
    def __init__(self, img_feature_channels: int):
        """Local statistique nerwork

        Args:
            img_feature_channels (int): [Number of input channels]
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=img_feature_channels, out_channels=512, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, concat_feature: torch.Tensor) -> torch.Tensor:
        x = self.conv1(concat_feature)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        local_statistics = self.conv3(x)
        return local_statistics


class GlobalStatisticsNetwork(nn.Module):
    """Global statistics network

    Args:
        feature_map_size (int): Size of input feature maps
        feature_map_channels (int): Number of channels in the input feature maps
        latent_dim (int): Dimension of the representationss
    """

    # self.global_stat_x = GlobalStatisticsNetwork(
    #     feature_map_size=self.backbone_feature_size,
    #     feature_map_channels=self.backbone_feature_channels,
    #     coding_id_channels=self.num_ch_id,
    #     coding_nonid_channels=self.num_ch_nonid,
    #     coding_size=self.coding_size,
    # )

    def __init__(
        self, feature_map_size: int, feature_map_channels: int, coding_channels: int, coding_size: int
    ):

        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            in_features=(feature_map_size ** 2 * feature_map_channels) * 2,
            out_features=512,
        )
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()

    def forward(
        self, feature_map: torch.Tensor, representation: torch.Tensor
    ) -> torch.Tensor:
        feature_map = self.flatten(feature_map)
        representation = self.flatten(representation)
        x = torch.cat([feature_map, representation], dim=1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        global_statistics = self.dense3(x)

        return global_statistics

#
# if __name__ == "__main__":
#
#     img_size = 128
#     x = torch.zeros((64, 3, img_size, img_size))
#     enc_sh = BaseEncoder(
#         img_size=img_size, in_channels=3, num_filters=16, kernel_size=1, repr_dim=64
#     )
#     enc_ex = BaseEncoder(
#         img_size=img_size,
#         in_channels=3,
#         num_filters=16,
#         kernel_size=1,
#         repr_dim=64,
#     )
#
#     sh_repr, sh_feature = enc_sh(x)
#     ex_repr, ex_feature = enc_ex(x)
#     merge_repr = torch.cat([sh_repr, ex_repr], dim=1)
#     merge_feature = torch.cat([sh_feature, ex_feature], dim=1)
#     concat_repr = tile_and_concat(tensor=merge_feature, vector=merge_repr)
#     t_loc = LocalStatisticsNetwork(img_feature_channels=concat_repr.size(1))
#     t_glob = GlobalStatisticsNetwork(
#         feature_map_size=merge_feature.size(2),
#         feature_map_channels=merge_feature.size(1),
#         latent_dim=merge_repr.size(1),
#     )
#     print(t_glob(feature_map=merge_feature, representation=merge_repr).shape)

