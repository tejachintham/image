import matplotlib.pyplot as plt

with open("data.txt") as f:
    content = f.readline().split("val")
    i=1
    for c in content:
        if(i==1):
            los=c.split(" ")
            i=i+1
        else:
            vallos=c.split(" ")
plt.plot(los)
plt.plot(vallos)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

            
    






