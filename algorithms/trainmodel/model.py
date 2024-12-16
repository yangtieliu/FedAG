import torch.nn as nn
import torch.nn.functional as F
from utils.model_config import CONFIGS_


class Net(nn.Module):
    def __init__(self, dataset='mnist', model='cnn'):
        super(Net, self).__init__()
        print("Creating model for {}".format(dataset))
        self.dataset = dataset
        configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim = CONFIGS_[dataset]
        print("Network configs:", configs)
        self.layers, self.layer_names, self.named_layers = self.build_network(
            configs, input_channel, self.output_dim)
        self.n_parameters = len(list(self.parameters()))  # self.parameters()来自nn.Module的子类
        # self.n_shared_parameters = len(list(self.get_learned_parameters()))

    def get_number_of_parameters(self):
        total_model_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_model_params

    def build_network(self, configs, input_channel, output_dim):  # output_dim设计冗余
        layers = nn.ModuleList()
        named_layers = {}
        layer_names = []
        kernel_size, stride, padding = 3, 2, 1
        for i, x in enumerate(configs):
            if x == 'F':
                layer_name = 'flatten{}'.format(i)
                layer = nn.Flatten(1)
                layers += [layer]
                layer_names += [layer_name]
            elif x == 'M':
                layer_name = 'maxpool{}'.format(i)
                pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                layers += [pool_layer]
                layer_names += [layer_name]
            else:
                cnn_name = 'encode_cnn{}'.format(i)  # 加密器
                cnn_layer = nn.Conv2d(input_channel, x, kernel_size=kernel_size, stride=stride, padding=padding)
                named_layers[cnn_name] = [cnn_layer.weight, cnn_layer.bias]

                bn_name = "encode_batchnorm{}".format(i)
                bn_layer = nn.BatchNorm2d(x)
                named_layers[bn_name] = [bn_layer.weight, bn_layer.bias]

                relu_name = "relu{}".format(i)
                relu_layer = nn.ReLU(inplace=True)

                layers += [cnn_layer, bn_layer, relu_layer]
                layer_names += [cnn_name, bn_name, relu_name]
                input_channel = x

        fc_layer_name1 = 'decode_fc1'
        fc_layer1 = nn.Linear(self.hidden_dim, self.latent_dim)
        layers += [fc_layer1]
        layer_names += [fc_layer_name1]
        named_layers[fc_layer_name1] = [fc_layer1.weight, fc_layer1.bias]

        fc_layer_name2 = 'decode_fc2'
        fc_layer2 = nn.Linear(self.latent_dim, self.output_dim)
        layers += [fc_layer2]
        layer_names += [fc_layer_name2]
        named_layers[fc_layer_name2] = [fc_layer2.weight, fc_layer2.bias]
        return layers, layer_names, named_layers

    def get_parameters_by_keyword(self, keyword='encode'):
        params = []
        for name, layer in zip(self.layer_names, self.layers):  # self.layers或self.named_layers
            if keyword in name:
                params += [layer.weight, layer.bias]
                # params += [self.named_layers[name]]
        return params
        # return [p for n, p in self.named_parameters() if keyword in n]

    def get_encoder(self):
        return self.get_parameters_by_keyword('encode')

    def get_decoder(self):
        return self.get_parameters_by_keyword('decode')

    def get_learned_parameters(self):
        return self.get_encoder() + self.get_decoder()

    def forward(self, x, start_layer_idx=0, logit=True):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        results = {}
        for layer in self.layers:
            x = layer(x)
        if self.output_dim > 1:
            results['output'] = F.log_softmax(x, dim=1)
        else:
            results['output'] = x
        if logit:
            results['logit'] = x
        return results

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        n_layers = len(self.layers)
        for layer_idx in range(n_layers + start_layer_idx, n_layers):
            layer = self.layers[layer_idx]
            #print(z.shape)
            z = layer(z)
        if self.output_dim > 1:
            out = F.log_softmax(z, dim=1)
        result = {'output': out}
        if logit:
            result['logit'] = z
        return result
