import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class LSTMAutoencoder(nn.Module):
    def __init__(self, sequence_length, n_features, prediction_time):
        super().__init__()

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.prediction_time = prediction_time

        # Encoder
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=100, batch_first=True)
        self.encoder2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)

        # Repeat vector for prediction_time
        self.repeat_vector = nn.Sequential(
            nn.ReplicationPad1d(padding=(0, prediction_time - 1)),
            nn.ReplicationPad1d(padding=(0, 0)),  # Adjusted padding
        )

        # Decoder
        self.decoder = nn.LSTM(input_size=50 + prediction_time - 1, hidden_size=100, batch_first=True)
        self.decoder2 = nn.LSTM(input_size=100, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        # Encoder
        # _, (x, _) = self.encoder(x)
        x, (_, _) = self.encoder(x)
        # output, (hn, cn) = rnn(x)
        x, (_, _) = self.encoder2(x)

        # Repeat vector for prediction_time
        x = self.repeat_vector(x)

        # Decoder
        x, (_, _) = self.decoder(x)
        x, (_, _) = self.decoder2(x)

        if self.prediction_time == 1:
            return x[:, -1, :].unsqueeze(dim=1)
        else:
            return x[:, -(self.prediction_time) :, :]


class MILClassifier(nn.Module):
    def __init__(self, input_dim=710, drop_p=0.0):
        super().__init__()
        # self.embedding = Temporal(input_dim, 512)
        # self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            # nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(drop_p),
            # nn.Linear(1024, 512),
            # # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(drop_p),
            nn.Linear(512, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.drop_p = drop_p
        self.weight_init()

    def weight_init(self):
        # for layer in self.classifier:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # x = self.embedding(x)
        # x = self.selfatt(x)

        x = x.view(-1, x.size(-1))

        x = self.classifier(x)

        return x


class MILandFE(nn.Module):
    def __init__(self, input_dim=710, drop_p=0.0):
        super().__init__()
        self.embedding = Temporal(input_dim, 512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0)
        self.classifier = nn.Sequential(
            # nn.Linear(input_dim, 512),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(drop_p),
            # nn.Linear(1024, 512),
            # # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(drop_p),
            nn.Linear(512, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.drop_p = drop_p
        self.weight_init()

    def weight_init(self):
        # for layer in self.classifier:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.embedding(x)
        x = self.selfatt(x)

        x = x.view(-1, x.size(-1))

        x = self.classifier(x)

        return x


class NormalHead(nn.Module):
    def __init__(self, in_channel=512, ratios=[16, 32], kernel_sizes=[1, 1, 1]):
        super(NormalHead, self).__init__()
        self.ratios = ratios
        # 기본값 [16, 32]
        self.kernel_sizes = kernel_sizes
        # 기본값 [1, 1, 1]

        self.build_layers(in_channel)

    def build_layers(self, in_channel):
        ratio_1, ratio_2 = self.ratios
        self.conv1 = nn.Conv1d(
            in_channel, in_channel // ratio_1, self.kernel_sizes[0], 1, self.kernel_sizes[0] // 2
        )
        # stride는 1, padding은 kernel_size // 2로 두면
        # (input_length - kernel_size + 2 * (kernel_size // 2)) + 1 == input_length
        # => 길이 유지
        self.bn1 = nn.BatchNorm1d(in_channel // ratio_1)
        self.conv2 = nn.Conv1d(
            in_channel // ratio_1, in_channel // ratio_2, self.kernel_sizes[1], 1, self.kernel_sizes[1] // 2
        )
        self.bn2 = nn.BatchNorm1d(in_channel // ratio_2)
        self.conv3 = nn.Conv1d(in_channel // ratio_2, 1, self.kernel_sizes[2], 1, self.kernel_sizes[2] // 2)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.bns = [self.bn1, self.bn2]

    def forward(self, x):
        """
        x: BN * C * T
        return BN * C // 64 * T and BN * 1 * T
        """
        outputs = []
        x = self.conv1(x)
        outputs.append(x)
        x = self.conv2(self.act(self.bn1(x)))
        outputs.append(x)
        x = self.sigmoid(self.conv3(self.act(self.bn2(x))))
        outputs.append(x)
        return outputs


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(2 * inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, d = x.size()
        qkvt = self.to_qkv(x).chunk(4, dim=-1)
        q, k, v, t = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkvt)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn1 = self.attend(dots)

        tmp_ones = torch.ones(n).cuda()
        tmp_n = torch.linspace(1, n, n).cuda()
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1, 1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.0)))
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b, self.heads, 1, 1)

        out = torch.cat([torch.matmul(attn1, v), torch.matmul(attn2, t)], dim=-1)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Temporal(nn.Module):
    # Temporal convolutional network
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x는 (batch * n crops, t snippets, d feature dim)
        # 영상 1개를 t개의 snippet(토막)으로 나누고 각 snippet은 d 차원 feature 벡터
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # permute를 하지 않고 conv1d를 하면 t가 채널축
        # => conv 특성상 1 entry 계산할 때 (t, 3)사이즈 필터를 곱해서 계산
        # => 이 entry는 각 snippet의 feature는 3개만 보지만 시간축(t)으로는 영상 전체를 보게 된다
        # => 마치 과거, 현재, 미래 정보를 다보고 계산하는 것과 마찬가지
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        x = x.permute(0, 2, 1)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # permute를 하게 되면 (batch * n crops, d feature dim, t snippets)
        # 이제 conv1d를 하면 (d, 3) 사이즈 필터를 곱해서 계산한다
        # => 바로 전, 현재, 바로 다음(또는 전전, 전, 현재) 시간의 영상 snippet 3개만 보고 각 snippet의 feature들은 전부 보게 된다
        # +@ 영상을 나누는 snippet(segment) 개수 유지(kernel_size=3, stride=1, padding=1)
        # +@ 이제 각 snippet의 feature dimension수를 조절 가능(필터 개수로 조절)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class WSAD(nn.Module):
    def __init__(
        self, input_size, ratio_sample=0.2, ratio_batch=0.4, ratios=[16, 32], kernel_sizes=[1, 1, 1]
    ):
        super().__init__()
        # self.flag = flag

        self.ratio_sample = ratio_sample
        # 기본값 0.2
        self.ratio_batch = ratio_batch
        # 기본값 0.4

        self.ratios = ratios
        # 기본값 [16, 32]
        self.kernel_sizes = kernel_sizes
        # 기본값 [1, 1, 1]

        self.normal_head = NormalHead(in_channel=512, ratios=ratios, kernel_sizes=kernel_sizes)
        self.embedding = Temporal(input_size, 512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0)
        # embedding + selfatt은 논문의 feature enhancer
        # embedding은 feature 차원을 permute + conv1d를 이용해 512로 변경
        # selfatt는 transformer계열 enhancer
        self.step = 0

    def get_normal_scores(self, x, ncrops=None):
        # x는 (batch * n crops, segment 개수, feature 차원 = 512(논문))
        new_x = x.permute(0, 2, 1)
        # conv1d에 넣기전에 (batch * n crops, feature 차원, segment 개수)로 변경

        outputs = self.normal_head(new_x)
        # normal_head는 conv1d - bn - relu - conv1d - bn - relu - conv1d - sig 3층 구조
        # outputs는 normal_head 안의 3개의 conv1d output을 담은 list (마지막 output은 conv1d + sig output)
        normal_scores = outputs[-1]
        xhs = outputs[:-1]

        if ncrops:
            b = normal_scores.shape[0] // ncrops
            normal_scores = normal_scores.view(b, ncrops, -1).mean(1)
            # (batch_size, t snippets)

        return xhs, normal_scores

    def get_mahalanobis_distance(self, feats, anchor, var, ncrops=None):
        # 첫번째는 feat는 (batch_size * n crops, 512 // 16, t snippets)
        # 두번째는 (batch_size * n crops, 512 // 32, t snippets)
        # BN은 각 feature(채널 축)별 batch*h*w개 평균, 분산 계산
        # => (b, c, h*w) -> (c)
        # => None으로 unsqueeze해서 (1, c, 1)로 변경
        distance = torch.sqrt(torch.sum((feats - anchor[None, :, None]) ** 2 / var[None, :, None], dim=1))
        # (x - m)^2/var -> torch.sum(dim=1)로 각배치 안의 각 토막(segment)별로 값 존재 (b, t)
        # sqrt후에도 사이즈 그대로 (b, t)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # var가 전부 1이면 distance는 BN running mean vector와 각 토막의 feature vector 간의 차이 벡터의 L2 norm 길이가 된다
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if ncrops:
            bs = distance.shape[0] // ncrops
            # b x t
            distance = distance.view(bs, ncrops, -1).mean(dim=1)
            # (batch_size, n crops, t snippets)을 dim=1로 평균 => 동일 영상 10개 crop들의 결과를 평균
            # => (batch_size, t snippets)
        return distance
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 배치 내의 각 영상의 각 토막 feature 벡터가
        # 데이터셋 분포 내 모든 영상의 모든 토막의 feature(512 // 16 또는 512 // 32 차원) 벡터들의 평균인 벡터(running_mean으로 추정)와
        # 얼마나 다른지 알려주는 mahalanobis 거리 계산
        # 데이터 분포내의 모든 토막 feature 벡터의 평균이고 정상토막의 비중이 이상토막의 비중보다 압도적으로 크기 때문에
        # 이 평균 벡터는 정상 토막의 기준처럼 사용 가능(anchor)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def pos_neg_select(self, feats, distance, ncrops):
        batch_select_ratio = self.ratio_batch
        # 기본값 0.4
        sample_select_ratio = self.ratio_sample
        # 기본값 0.2
        bs, c, t = feats.shape
        # 첫번째는 (batch_size * n crops, 512 // 16, t snippets)
        # 두번째는 (batch_size * n crops, 512 // 32, t snippets)
        select_num_sample = int(t * sample_select_ratio)
        # sample-level selection(SLS)은 20%
        select_num_batch = int(bs // 2 * t * batch_select_ratio)
        # 데이터는 torch.cat((정상영상, 이상영상), dim=0)으로 정상영상 배치 뒤에 이상영상 배치가 붙어있음
        # => bs // 2가 실제 batch_size * n crops 개수
        # => batch-level selection(BLS)은 (bs // 2) * t개 중 40%
        # ==> 40 // 2 해서 사실상 SLS와 동일 비율로 배치 하나당 20%

        feats = feats.view(bs, ncrops, c, t).mean(dim=1)  # b x c x t
        # 동일 영상에서 나온 10개 crop들 결과 평균
        # => (batch_size, c features, t snippets)
        nor_distance = distance[: bs // 2]  # b x t
        # distance는 10개 crop들을 이미 평균내고 (batch_size, t snippets)
        # 그리고 배치 앞 절반은 정상영상 배치 => (n_batch_size = batch_size // 2, t snippets)
        nor_feats = feats[: bs // 2].permute(0, 2, 1)  # b x t x c
        # 정상부분 앞 절반만 가져와 permute => (n_batch_size, t snippets, c features)
        abn_distance = distance[bs // 2 :]  # b x t
        # 배치 뒤 절반은 이상영상 배치 (a_batch_size = batch_size // 2, t snippets)
        abn_feats = feats[bs // 2 :].permute(0, 2, 1)  # b x t x c
        # (a_batch_size, t snippets, c features)
        abn_distance_flatten = abn_distance.reshape(-1)
        # (a_batch_size * t snippets)
        abn_feats_flatten = abn_feats.reshape(-1, c)
        # (a_batch_size * t snippets, c features)

        mask_select_abnormal_sample = torch.zeros_like(abn_distance, dtype=torch.bool)
        # (a_batch_size, t snippets)
        topk_abnormal_sample = torch.topk(abn_distance, select_num_sample, dim=-1)[1]
        # torch.topk(abn_distance, select_num_sample, dim=-1)는 top k개 value와 그 value들 indices를 담고 있다
        # value와 indices 둘 다 (a_batch_size, top K = select_num_sample) 형태
        # => [1]로 indices만 가져오기
        mask_select_abnormal_sample.scatter_(
            dim=1,
            index=topk_abnormal_sample,
            src=torch.full_like(topk_abnormal_sample, True, dtype=torch.bool),
        )
        # (a_batch_size, t snippets) 형태이고 True는 a_batch_size * select_num_sample개이고 나머지는 False
        # (top k에 속하는 index 자리만 True, 나머지는 False)
        # scatter는 gather의 reverse operation

        mask_select_abnormal_batch = torch.zeros_like(abn_distance_flatten, dtype=torch.bool)
        # (a_batch_size * t snippets)
        topk_abnormal_batch = torch.topk(abn_distance_flatten, select_num_batch, dim=-1)[1]
        # (a_batch_size * select_num_batch)
        # top K = select_num_batch 개 indices
        mask_select_abnormal_batch.scatter_(
            dim=0,
            index=topk_abnormal_batch,
            src=torch.full_like(topk_abnormal_batch, True, dtype=torch.bool),
        )
        # (a_batch_size * t snippets)

        mask_select_abnormal = mask_select_abnormal_batch | mask_select_abnormal_sample.reshape(-1)
        # SLS와 BLS를 or 연산 | 으로 합쳐서 논문의 Sample-Batch Selection(SBS)
        select_abn_feats = abn_feats_flatten[mask_select_abnormal]
        # mask_select_abnormal는 (a_batch_size * t snippets)개 중 num_select_abnormal개만 True고 나머진 False
        # abn_feats_flatten의 (a_batch_size * t snippets, c features)에서 mask_select_abnormal를 indices로 쓰면
        # (num_select_abnormal, c feature) 형태가 된다
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # select_abn_feats는 SLS와 BLS를 합쳐 SBS를 만드는 과정에서 상위 ~%에 들었다는 정보만 남고 distance 상위 몇번째인지 순서 정보가 날아간다
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        num_select_abnormal = torch.sum(mask_select_abnormal)
        # SBS 추출 개수

        k_nor = int(num_select_abnormal / (bs // 2)) + 1
        # 이상영상 배치에서 SBS로 선택한 개수 / 배치내 영상 개수 == 1 배치 당 평균 선택 개수
        # + 1을 해주어서 정상 영상에서 선택된 토막(snippets)개수가 이상 영상 선택 토막 개수보다 크게 설정
        topk_normal_sample = torch.topk(nor_distance, k_nor, dim=-1)[1]
        # nor_distance는 (n_batch_size, t snippets)
        # topk_normal_sample는 각 영상의 t개 토막 중 상위 k_nor개의 indices
        # => (n_batch_size, k_nor)
        select_nor_feats = torch.gather(nor_feats, 1, topk_normal_sample[..., None].expand(-1, -1, c))
        # nor_feats는 (n_batch_size, t snippets, c features)
        # gather의 index는 input과 차원수가 같아야하므로 None으로 (n_batch_size, k_nor, 1), expand로 (n_batch_size, k_nor, c) 형태로 변경
        # expand : Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
        # gather dimension이 1 => select_nor_feats[i][j][k] = nor_feats[i][topk_normal_sample[i][j][k]][k]
        # select_nor_feats는 (n_batch_size, k_nor, c) 형태 (gather는 index와 output 형태가 동일)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # select_abn_feats와 다르게 select_nor_feats는 크기 순서를 지우지 않고 gather를 써서 dim=1 방향으로 nor_distance 값 내림차순
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        select_nor_feats = select_nor_feats.permute(1, 0, 2).reshape(-1, c)
        # (k_nor, n_batch_size, c)로 바꾼 후 reshape로 (k_nor * n_batch_size, c) 형태
        select_nor_feats = select_nor_feats[:num_select_abnormal]
        # k_nor * n_batch_size는 num_select_abnormal보다 크다 => out of index 에러 안 일어남
        # select_nor_feats는 최종적으로 (num_select_abnormal, c feature) 형태

        return select_nor_feats, select_abn_feats

    def forward(self, x, flag="Eval"):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            # 실험에 사용한 I3D UCF-Crime feature는 하나의 영상을 중앙, 4코너 + 중앙, 4코너 거울상 = 10개 crop으로 증강해서 계산
            # => batch 개수, n crop 개수, t 토막(snippet, segment) 개수, d snippet당 feature 차원수

            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)
        # feature enhancer를 지난 feature의 차원수 d == 512(논문)

        normal_feats, normal_scores = self.get_normal_scores(x, n)
        # normal_head는 conv1d - bn - relu - conv1d - bn - relu - conv1d - sig 3층 구조
        # normal_feats는 [첫 conv1d output, 두번째 conv1d output]
        # => (batch_size * n crops, 512 // 16, t snippets), (batch_size * n crops, 512 // 32, t snippets) 형태
        # normal_scores는 마지막 conv1d - sig output => (batch_size, t snippets) 형태 (n crops는 평균-> 1)

        anchors = [bn.running_mean for bn in self.normal_head.bns]
        variances = [bn.running_var for bn in self.normal_head.bns]
        # conv1d output 바로 뒤 bn은 conv1d output 전체 분포 추정 평균, 분산을 담고 있다
        # 두개의 bn => 첫 conv1d output, 두번째 conv1d output 추정 평균, 분산

        distances = [
            self.get_mahalanobis_distance(normal_feat, anchor, var, ncrops=n)
            for normal_feat, anchor, var in zip(normal_feats, anchors, variances)
        ]
        # list안의 각 distance는 (batch_size, t snippets) 형태

        if flag == "Train":

            select_normals = []
            select_abnormals = []
            for feat, distance in zip(normal_feats, distances):
                select_feat_normal, select_feat_abnormal = self.pos_neg_select(feat, distance, n)
                # select_feat_normal, select_feat_abnormal 둘다 (num_select_abnormal, c feature) 형태
                select_normals.append(select_feat_normal[..., None])
                select_abnormals.append(select_feat_abnormal[..., None])
                # 두 정상, 이상 리스트 모두 feature 두개씩
                # 첫번째는 (num_select_abnormal, 512 // 16 feature, 1)
                # 두번째는 (num_select_abnormal, 512 // 32 feature, 1)

            bn_results = dict(
                anchors=anchors,
                variances=variances,
                select_normals=select_normals,
                select_abnormals=select_abnormals,
            )
            # breakpoint()
            distance_sum = sum(distances)

            return {
                "pre_normal_scores": normal_scores[0 : b // 2],
                # classifier 학습에 사용되는 normal loss 계산에는 label 노이즈가 없는 normal 영상만 사용
                # (label noise: MIL은 비디오 단위 라벨링만 있음
                # => 이상 영상안의 normal snippet을 abnormal snippet으로 판단 하는 등의 noise 발생 가능)
                # 정상 영상의 snippet들은 무조건 정상 => 정상 영상 하나의 t snippets의 scores => t 차원 score 벡터
                # ==> 이 t 차원 score 벡터의 L2 norm 값 * n_batch_size 개 정상 영상 == normal loss
                # ==> L2 norm인 normal loss가 작아지기 위해서 정상 영상 snippet들의 예측 score가 작아지는 방향으로 학습
                # 논문 3.4 확인
                "bn_results": bn_results,
                # mpp loss 계산에 사용
                # 논문 3.2 확인
                # @@@@@@@@@@@@@@@@@@@@@@@@@
                # bce loss를 위해 추가
                "normal_scores": normal_scores,
                "scores": distance_sum * normal_scores,
            }
        elif flag == "Train_extra":
            distance_sum = sum(distances)
            # (batch_size, t snippets) 형태인 distance들 sum

            return {
                "normal_scores": normal_scores,
                "scores": distance_sum * normal_scores,
            }
        elif flag == "Eval_MPP":

            distance_sum = sum(distances)
            # (batch_size, t snippets) 형태인 distance들 sum

            return {
                "normal_scores": normal_scores,
                "scores": distance_sum * normal_scores,
            }
        else:

            distance_sum = sum(distances)
            # (batch_size, t snippets) 형태인 distance들 sum

            return distance_sum * normal_scores
            # normal_scores도 (batch_size, t snippets) 형태
