def train(device, dataloaders, dataset_sizes, learning_rate, num_epochs, early_stop_patience, model_path, pre_trained_baseline_net):
    cvae_net = CVAE(200, 500, 500, pre_trained_baseline_net).to(device)
    optimizer = optim.Adam(cvae_net.parameters(), lr=learning_rate)
    best_loss = np.inf
    early_stop_count = 0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    val_inp, digits = get_val_images(num_quadrant_inputs=1, num_images=30, shuffle=False)
    val_inp = val_inp.to(device)
    samples = []
    losses = []

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            running_loss = 0.0
            bar = tqdm(dataloaders[phase], desc=f'CVAE Epoch {epoch} {phase}'.ljust(20))
            for i, batch in enumerate(bar):
                inputs = batch['input'].to(device)
                outputs = batch['output'].to(device)

                if phase == 'train':
                    cvae_net.train()
                    optimizer.zero_grad()
                    loc, recon_loc, scale, zs = cvae_net(inputs, outputs)
                    loss = cvae_net.compute_loss(inputs, outputs, loc, scale, zs, recon_loc)
                    loss.backward()
                    optimizer.step()
                else:
                    cvae_net.eval()
                    with torch.no_grad():
                        loc, recon_loc, scale, zs = cvae_net(inputs, outputs)
                        loss = cvae_net.compute_loss(inputs, outputs, loc, scale, zs, recon_loc)

                running_loss += loss.item()
                if i % 10 == 0:
                    bar.set_postfix(loss='{:.2f}'.format(loss.item()), early_stop_count=early_stop_count)

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    cvae_net.save(model_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    cvae_net.load(model_path)
    samples = pd.concat(samples, axis=0, ignore_index=True)
    samples.to_csv('samples.csv', index=False)

    losses = pd.concat(losses, axis=0, ignore_index=True)
    losses.to_csv('losses.csv', index=False)

    return cvae_net


def predict_samples(inputs, digits, pre_trained_cvae, epoch_frac):
    pre_trained_cvae.eval()
    with torch.no_grad():
        loc, recon_loc, scale, zs = pre_trained_cvae(inputs)
        y_loc = recon_loc.detach().cpu().numpy()
        dfs = pd.DataFrame(data=y_loc)
        dfs['digit'] = digits.numpy()
        dfs['epoch'] = epoch_frac
    return dfs
