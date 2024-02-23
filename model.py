import transformers
from torch import nn
from transformers.models.bert.modeling_bert import BertModel, BertLMPredictionHead
import torch


class AntibodyFormer(nn.Module):
    def __init__(self, args):
        super(AntibodyFormer, self).__init__()
        config = transformers.BertConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=args.max_position_embeddings,
            hidden_dropout_prob=args.hidden_dropout_prob,
        )

        self.AntiFormer = BertModel(config)
        self.linear_transformation = nn.Linear(config.hidden_size, 32)

        self.feedforward = nn.Linear(2560, 32)

        self.classfier = nn.Linear(32, 2)
        self.gcn = MultiChannelGCN(input_dim=512, hidden_dim=512, output_dim=2, num_channels=3)


    def forward(self,input_ids, attention_mask, token_type_ids, node_map, adjs):
        outputs = self.AntiFormer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        sequence_output, pooled_output = outputs[:2]
        cls_f = torch.cat([outputs['hidden_states'][-i][:, 0] for i in range(1, 9)], dim=-1)
        hyper_adjs = self.gcn(node_map, adjs)
        KG_prob = torch.argmax(hyper_adjs, dim=1, keepdim=True)
        KG_conf = torch.max(hyper_adjs, dim=1, keepdim=True)

        x = self.linear_transformation(sequence_output)  # x: bs*Lx32
        x = x.mean(dim=2)
        x = torch.cat((cls_f, x), dim=1)
        x = self.feedforward(x)
        x = self.classfier(x)
        x = x * hyper_adjs

        return x





class MultiChannelGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_channels):
        super(MultiChannelGCN, self).__init__()
        self.gc_layers = nn.ModuleList()
        for _ in range(num_channels):
            self.gc_layers.append(GraphConvolution(input_dim, hidden_dim))
        self.hyper_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjs):
        channel_outputs = []
        for i in range(len(adjs)):
            channel_output = self.gc_layers[i](x[i], adjs[i])
            channel_output = torch.relu(channel_output)
            channel_outputs.append(channel_output)

        combined_output = torch.stack(channel_outputs, dim=0).sum(dim=0)

        hyper_adjs = self.hyper_layer(combined_output)
        return hyper_adjs


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output


