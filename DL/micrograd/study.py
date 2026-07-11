from nn import *

random.seed(42)

X = []
y = []
N = 500
R = 0.7

for _ in range(N):
    x1 = random.uniform(-1.0, 1.0)
    x2 = random.uniform(-1.0, 1.0)
    X.append([x1, x2])

    if x1 ** 2 + x2 ** 2 < R ** 2:
        y.append([1.0])
    else:
        y.append([-1.0])

model = MLP(2, [8, 1])

epochs = 1000
learning_rate = 0.02

for epoch in range(epochs):
    ypred = [model(x) for x in X]

    loss = sum((yp - yt[0]) ** 2 for yt, yp in zip(y, ypred)) * (1.0 / N)

    model.zero_grad()

    loss.backward()

    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0 or epoch == epochs - 1:
        correct = 0
        for yt, yp in zip(y, ypred):
            if (yp.data > 0 and yt[0] > 0) or (yp.data <= 0 and yt[0] < 0):
                correct += 1
        accuracy = (correct / N) * 100
        print(f"Эпоха {epoch:3d} | Ошибка (Loss): {loss.data:.4f} | Точность: {accuracy:.1f}%")


print("\n--- Карта предсказаний нейросети в пространстве ---")

for x2 in sorted([i*0.15 for i in range(-7, 8)], reverse=True):
    row = ""
    for x1 in [i*0.10 for i in range(-10, 11)]:
        pred = model([x1, x2]).data
        if pred > 0:
            row += "."
        else:
            row += "X"
    print(row)