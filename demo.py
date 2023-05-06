import torch

a = torch.load(r"C:\Users\Wei\Desktop\segformer_b2_backbone_weights.pth")

weights_dict = {}
for k, v in a.items():
    new_k = "module.backbone."+k
    weights_dict[new_k] = v
torch.save(weights_dict,r"C:\Users\Wei\Desktop\backbone_weights.pth")

# aa = b["model_G_state_dict"]['module.Tenc_x2.patch_embed1.proj.weight']
print()


print()