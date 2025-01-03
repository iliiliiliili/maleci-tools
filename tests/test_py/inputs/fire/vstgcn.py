"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from variational import VariationalBase, VariationalConvolution, init_weights as vnn_init_weights
import variational
from typing import Any, List, Optional, Literal, Tuple, Union


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def weights_init(module_, bs=1):
    if isinstance(module_, nn.Conv2d) and bs == 1:
        nn.init.kaiming_normal_(module_.weight, mode="fan_out")
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Conv2d) and bs != 1:
        nn.init.normal_(
            module_.weight,
            0,
            math.sqrt(
                2.0
                / (
                    module_.weight.size(0)
                    * module_.weight.size(1)
                    * module_.weight.size(2)
                    * bs
                )
            ),
        )
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.BatchNorm2d):
        nn.init.constant_(module_.weight, bs)
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Linear):
        nn.init.normal_(module_.weight, 0, math.sqrt(2.0 / bs))


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_):
        super(GraphConvolution, self).__init__()
        self.cuda_ = cuda_
        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.graph_attn, 1)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = 3
        self.g_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1))
            weights_init(self.g_conv[i], bs=self.num_subset)

        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
            weights_init(self.gcn_residual[0], bs=1)
            weights_init(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        weights_init(self.bn, bs=1e-6)

    def forward(self, x):
        N, C, T, V = x.size()
        if self.cuda_:
            A = self.A.cuda(x.get_device())
        else:
            A = self.A
        A = A * self.graph_attn
        hidden_ = None
        for i in range(self.num_subset):
            x_a = x.view(N, C * T, V)
            z = self.g_conv[i](torch.matmul(x_a, A[i]).view(N, C, T, V))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        hidden_ += self.gcn_residual(x)
        return hidden_


class VariationalTemporalConvolution(VariationalConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        global_std_mode="none",
    ):
        
        pad = int((kernel_size - 1) / 2)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(pad, 0),
            batch_norm_mode="mean+std",
            use_batch_norm=True,
            activation=None,
            activation_mode="none",
            global_std_mode=global_std_mode
        )


class VariationalGraphConvolution(VariationalBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A,
        cuda_,
        activation_mode="mean",
        global_std_mode="none",
    ) -> None:
        super().__init__()

        means = GraphConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            A=A,
            cuda_=cuda_,
        )

        if global_std_mode == "replace":
            stds = None
        else:
            stds = GraphConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                A=A,
                cuda_=cuda_,
            )

        super().build(
            means,
            stds,
            None,
            None,
            activation=nn.ReLU(),
            activation_mode=activation_mode,
            use_batch_norm=False,
            batch_norm_mode=None,
            global_std_mode=global_std_mode,
        )
    def _init_weights(self):

        all_submodules = [
            lambda x: (x.g_conv[0].weight, True),
            lambda x: (x.g_conv[0].bias, False),
            lambda x: (x.g_conv[1].weight, True),
            lambda x: (x.g_conv[1].bias, False),
            lambda x: (x.g_conv[2].weight, True),
            lambda x: (x.g_conv[2].bias, False),
            lambda x: (x.gcn_residual[0].weight if isinstance(x.gcn_residual, torch.nn.Sequential) else None, True),
            lambda x: (x.gcn_residual[0].bias if isinstance(x.gcn_residual, torch.nn.Sequential) else None, False),
        ]

        vnn_init_weights(self, all_submodules)


class VariationalStgcnBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, A, cuda_=False, stride=1, residual=True, **kwargs
    ):
        super().__init__()

        self.gcn = VariationalGraphConvolution(in_channels, out_channels, A, cuda_, **kwargs)
        self.tcn = VariationalTemporalConvolution(out_channels, out_channels, stride=stride, **kwargs)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = VariationalTemporalConvolution(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x):

        result = self.gcn(x)
        result = self.tcn(result)

        result += self.residual(x)
        result = self.relu(result)
        
        return result


class VStgcn(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        cuda_=True,
        FIX_GAUSSIAN=None,
        INIT_WEIGHTS="usual",
        samples=4,
        test_samples=4,
        **kwargs
    ):
        super().__init__()

        self.default_samples = samples
        self.test_samples = test_samples

        VariationalBase.FIX_GAUSSIAN = FIX_GAUSSIAN
        VariationalBase.INIT_WEIGHTS = INIT_WEIGHTS

        if VariationalBase.FIX_GAUSSIAN is not None:
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        weights_init(self.data_bn, bs=1)

        self.layers = nn.ModuleDict(
            {
                "layer1": VariationalStgcnBlock(in_channels, 64, A, cuda_, residual=False, **kwargs),
                "layer2": VariationalStgcnBlock(64, 64, A, cuda_, **kwargs),
                "layer3": VariationalStgcnBlock(64, 64, A, cuda_, **kwargs),
                "layer4": VariationalStgcnBlock(64, 64, A, cuda_, **kwargs),
                "layer5": VariationalStgcnBlock(64, 128, A, cuda_, stride=2, **kwargs),
                "layer6": VariationalStgcnBlock(128, 128, A, cuda_, **kwargs),
                "layer7": VariationalStgcnBlock(128, 128, A, cuda_, **kwargs),
                "layer8": VariationalStgcnBlock(128, 256, A, cuda_, stride=2, **kwargs),
                "layer9": VariationalStgcnBlock(256, 256, A, cuda_, **kwargs),
                "layer10": VariationalStgcnBlock(256, 256, A, cuda_, **kwargs),
            }
        )

        self.fc = nn.Linear(256, num_class)
        weights_init(self.fc, bs=num_class)

    def forward(self, x, samples=None, combine_predictions=True):

        if samples is None:
            if self.training:
                samples = self.default_samples
            else:
                samples = self.test_samples

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )

        outputs = []

        for s in range(samples):

            current_x = x

            for i in range(len(self.layers)):
                current_x = self.layers["layer" + str(i + 1)](current_x)
            # N*M,C,T,V
            c_new = current_x.size(1)
            current_x = current_x.view(N, M, c_new, -1)
            current_x = current_x.mean(3).mean(1)
            current_x = self.fc(current_x)
            outputs.append(current_x)
        
        result_var, result = torch.var_mean(torch.stack(outputs, dim=0), dim=0, unbiased=False)

        return result #, result_var


class VVV(VStgcn):
    def __init__(self):
        super()
        
class VVV2(nn.Module, VVV, variational.VariationalBase):
    def __init__(self):
        super()

class VVV3(VariationalBase, torch.nn.Module):
    def __init__(self):
        super()

c = VStgcn(12)

if "__main__" == __name__:
    print("Lol")