import matplotlib.pyplot as plt
import pickle

alphas = [10, 1, 0.1, 0.01, 0.001]
plt.figure()
for alpha in alphas:
    params = {
        'tau': -1, 'k': 4, 'alpha': alpha, 'lambda_': 1e-6, 'max_epoch': 30,
    }
    save_name = f"model{'_'.join(map(str, params.values()))}"

    with open(f'../output/{save_name}.pickle', 'rb') as f:
        model = pickle.load(f)
    history = [4.27355490]
    history.extend(model._training_history)
    plt.plot(history)

plt.legend(alphas)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('../results/finding_alpha.pdf')
