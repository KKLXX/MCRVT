import torch
import torch.optim as optim
from MCRVT.Datasets.Datapre.dataload import get_train_loader, get_test_loader
from modules.MCRVTLoss import MCRVTLoss
from evaluators.MCRVT_evaluators import evaluate
from Joint_networks.Joint_Net import JointNetwork
from Joint_networks.Training import train

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointNetwork(num_classes=7).to(device)
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    criterion = MCRVTLoss(
        alpha=1.0,
        beta=0.1,
        lambda_=0.5,
        xi_o=0.01, xi_alpha=0.1, xi_beta=0.05,
        num_scales=3,
        num_classes=7
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    num_epochs = 500
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: ")
            print(f"  Total Loss: {train_loss:.4f}")
            print(f"  Components: {loss_dict}")
        print(
            f"Epoch {epoch}: "
            f"WF1 = {test_metrics['wf1']:.4f}, "
            f"F1 = {test_metrics['f1']:.4f}"
        )
if __name__ == "__main__":
    main()