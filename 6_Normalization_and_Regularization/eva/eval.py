def fit(model, device, epochs, train, test, train_loader, test_loader, optimizer, learning_rate, momentum, weight_decay):
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list=[]
    test_accuracy_list=[]

    optimizer = optimizer(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_loss,train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss,test_acc = test(model, device, test_loader)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        train_accuracy_list.append(train_acc)
        test_accuracy_list.append(test_acc)

    return train_accuracy_list, test_accuracy_list, train_loss_list, test_loss_list
