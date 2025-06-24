@app.route("/api/train", methods=["POST"])
def train_model():
    dataset = request.files.get("dataset")
    user_id = request.form.get("user_id")  # Unique identifier for user
    task = request.form.get("task")  # Optional: specify task to fine-tune

    # Save dataset to temporary storage
    dataset_path = f"data/{user_id}/dataset"
    os.makedirs(dataset_path, exist_ok=True)
    dataset.save(os.path.join(dataset_path, "dataset.csv"))

    # Initialize model
    model = Labelee(vocab_size=10000, feature_dim=768, num_classes=1000)
    checkpoint_path = f"checkpoints/{user_id}/model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))  # Load user’s previous weights

    # Set up dataset and training
    dataset = VisionLanguageDataset(dataset_path, transform=...)  # From your May 26, 2025 conversation
    dataloader = DataLoader(dataset, batch_size=32, num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = MultiTaskLoss(alpha=1.0, beta=0.5, gamma=0.3)

    # Train model
    for epoch in range(5):  # Limited epochs for fine-tuning
        for batch in dataloader:
            images, input_ids, attention_mask, labels = batch
            outputs = model(images, input_ids, attention_mask, task=task)
            loss = criterion(outputs, labels, task=task)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save updated model
    torch.save(model.state_dict(), checkpoint_path)
    return jsonify({"status": "Training completed", "checkpoint": checkpoint_path})