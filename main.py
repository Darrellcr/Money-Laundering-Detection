import torch


def main():
    if torch.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(5, device=mps_device)
        print(x)


if __name__ == "__main__":
    main()
