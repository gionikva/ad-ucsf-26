from pathlib import Path
import sys

# Force the project root (parent of tests/) into the search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from utils.dataset import ADNISegDataset
from models.msr_cog import MSRCog


def test_msr_cog():

    dataset = ADNISegDataset(split="train")
    image = dataset[0]["mri"].to("cuda")
    model = MSRCog(
        "./weights/lightmedseg/downsampled/best.pth",
        input_size=(1, 160, 160, 160),
        demo_size=5,
        out_size=1,
    ).to("cuda")
    res = model(
        image.unsqueeze(dim=0), torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(dim=0).to("cuda")
    )
    
    print(res)
    


def main():
    test_msr_cog()


if __name__ == "__main__":
    main()
