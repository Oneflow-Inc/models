from vit import ViT_B_16_224, ViT_B_16_384, ViT_B_32_224, ViT_B_32_384, ViT_L_16_224, ViT_L_32_224

def build_model(args):
    if args.model_arch == "vit_b_16_224":
        return ViT_B_16_224(args)
    elif args.model_arch == "vit_b_16_384":
        return ViT_B_16_384(args)
    elif args.model_arch == "vit_b_32_224":
        return ViT_B_32_224(args)
    elif args.model_arch == "vit_b_32_384":
        return ViT_B_32_384(args)
    elif args.model_arch == "vit_l_16_224":
        return ViT_L_16_224(args)
    elif args.model_arch == "vit_l_32_224":
        return ViT_L_32_224(args)
