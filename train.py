import torch
from tqdm.auto import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs=50,
    patience=7,
    best_model_path="models/best_custom_cnn_htr.pth",
    backup_path="models/final_training_backup.pth"
):

    # Tracking
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):

        # ==========================================
        # TRAINING
        # ==========================================
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
        )

        for batch in progress_bar:

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # (B, T, C) -> (T, B, C)
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)

            batch_size = images.size(0)
            time_steps = outputs.size(1)

            input_lengths = torch.full(
                (batch_size,),
                time_steps,
                dtype=torch.long,
                device=device
            )

            # Flatten only valid characters for CTC loss
            targets = torch.cat([
                labels[i, :label_lengths[i]]
                for i in range(labels.size(0))
            ])

            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                label_lengths
            )

            # Backpropagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # ==========================================
        # VALIDATION
        # ==========================================
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():

            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"
            ):

                images = batch["images"].to(device)
                labels = batch["labels"].to(device)
                label_lengths = batch["label_lengths"].to(device)

                outputs = model(images)

                log_probs = outputs.log_softmax(2).permute(1, 0, 2)

                batch_size = images.size(0)
                time_steps = outputs.size(1)

                input_lengths = torch.full(
                    (batch_size,),
                    time_steps,
                    dtype=torch.long,
                    device=device
                )

                targets = torch.cat([
                    labels[i, :label_lengths[i]]
                    for i in range(labels.size(0))
                ])

                loss = criterion(
                    log_probs,
                    targets,
                    input_lengths,
                    label_lengths
                )

                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # Update learning rate based on validation loss
        scheduler.step(epoch_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ==========================================
        # SAVE BEST MODEL
        # ==========================================
        if epoch_val_loss < best_val_loss:

            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
            }, best_model_path)

            print("✓ New best model saved!")

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement for "
                f"{epochs_without_improvement}/{patience} epochs"
            )

        # ==========================================
        # EARLY STOPPING
        # ==========================================
        if epochs_without_improvement >= patience:

            print("\nEarly stopping triggered!")
            break

    # ==========================================
    # FINAL TRAINING BACKUP
    # ==========================================
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, backup_path)

    print("Final training backup saved!")

    return train_losses, val_losses, best_val_loss