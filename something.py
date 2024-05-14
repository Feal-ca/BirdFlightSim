import matplotlib.pyplot as plt

# Assuming these are your data
angles = [0, 10, 20, 45, 90]
lift1 = [32.90, 85.61, 103.61, 92.37, 0.11]
drag1 = [10.83, 20.18, 39.36, 89.84, 153.10]
LD1 = [3.04,4.24,2.63,1.03,0.00]

lift2 = [30.73, 101.54, 113.87, 107.12, 0.11]
drag2 = [8.74, 22.47, 47.26, 106.99, 168.03]
LD2 = [3.52,4.52,2.41,1.00,0.00]

plt.figure(figsize=(10,6))
plt.plot(angles, lift1, label='Lift', marker='o')
plt.plot(angles, drag1, label='Drag', marker='o')
plt.xlabel('Angles [Graus]')
plt.ylabel('Força [N]')
plt.title('Lift i Drag a diferents angles')
plt.legend()
plt.grid(True)
plt.savefig("LiftDrag1.png")


plt.figure(figsize=(10,6))
plt.plot(angles, lift2, label='Lift', marker='o')
plt.plot(angles, drag2, label='Drag', marker='o')
plt.xlabel('Angles [Graus]')
plt.ylabel('Força [N]')
plt.title('Lift i Drag a diferents angles')
plt.legend()
plt.grid(True)
plt.savefig("LiftDrag2.png")

plt.figure(figsize=(10,6))
plt.plot(angles, LD1, label='Ala 1', marker='o')
plt.plot(angles, LD2, label='Ala 2', marker='o')
plt.xlabel('Angles [Graus]')
plt.ylabel('Lift/Drag')
plt.title('L/D a diferents angles')
plt.legend()
plt.grid(True)
plt.savefig("LD.png")
