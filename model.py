import os
import torch
from torch.fx.experimental.proxy_tensor import make_fx


class DummyModel(torch.nn.Module):
    def __init__(self, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys
        super(DummyModel, self).__init__()
    
    def forward(self, pos, edge_index, edge_cell_shift, cell, atom_types):
        # Dummy operations to produce outputs
        
        # Total energy: sum of all position values, reshaped to (1,1)
        total_energy = pos.sum().reshape(1, 1)
        
        forces = pos
        
        # Atomic energy: sum along last dimension of positions, reshaped to (64,1)
        atomic_energy = pos.sum(dim=-1, keepdim=True)
        
        # Virial: outer product of first three positions and cell, reshaped to (1,3,3)
        virial = torch.einsum('ij,kl->ijkl', pos[:3], cell).sum(dim=(0, 1)).reshape(1, 3, 3)
        
        return total_energy, forces, atomic_energy, virial
    

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # inputs:
        #   pos (64,3)
        #   edge_index (2,264)
        #   edge_cell_shift (264,3)
        #   cell (3,3)
        #   atom_types (64,1)
    # outputs:
        #   total_energy (1,1)
        #   forces (64,3)
        #   atomic_energy (64,1)
        #   virial (1,3,3)

    data = (
        torch.randn(64,3).to(device=device),
        torch.randn(2,64).to(device=device),
        torch.randn(264,3).to(device=device),
        torch.randn(3,3).to(device=device),
        torch.randn(64,1).to(device=device)
    )
    output_keys = ['total_energy', 'forces', 'atomic_energy', 'virial']
    input_keys = ['pos', 'edge_index', 'edge_cell_shift', 'cell', 'atom_types']
        
    p = DummyModel(input_keys, output_keys)
    pgm = make_fx(p)(*data)

    so_path = torch._export.aot_compile(
                pgm,
                data,
                options={"aot_inductor.output_path": os.path.join(os.getcwd(), "model.so"),
                        # "triton.cudagraph_trees": True,
                        # "max_autotune": True,
                        # "coordinate_descent_tuning": True
            })

