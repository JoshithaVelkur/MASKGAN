import os
import torch
import sys

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device("cuda" if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else "cpu")
        self.Tensor = torch.FloatTensor if self.device.type == "cpu" else torch.cuda.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label):
        save_filename = f'{epoch_label}_net_{network_label}.pth'
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.to('cpu').state_dict(), save_path)
        # Restore network to device after saving
        network.to(self.device)

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = f'{epoch_label}_net_{network_label}.pth'
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        print(save_filename)

        if not os.path.isfile(save_path):
            print(f'{save_path} not exists yet!')
            if network_label == 'G':
                raise RuntimeError('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path, map_location=self.device, weights_only=True))
            except:
                pretrained_dict = torch.load(save_path, map_location=self.device, weights_only=True)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(f'Pretrained network {network_label} has excessive layers; Only loading layers that are used.')
                except:
                    print(f'Pretrained network {network_label} has fewer layers; The following are not initialized:')
                    for k, v in pretrained_dict.items():
                        if k in model_dict and v.size() == model_dict[k].size():
                            model_dict[k] = v

                    not_initialized = set()
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

        # Move network to device after loading
        network.to(self.device)

    def update_learning_rate(self):
        pass