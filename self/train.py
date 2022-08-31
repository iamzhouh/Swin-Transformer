import os.path
import sys

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import torch.optim

sys.path.append("..")

from dataset_process import FlowerDataset
from model import swin_tiny_patch4_window7_224

batch_size = 24
datasets_dir = "/newdisk/zhouh/flower_photos"

if os.path.exists('./model') is False:
    os.makedirs('./model')

creat_file = open('log.txt','a+')

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 数据集http://download.tensorflow.org/example_images/flower_photos.tgz
train_dataset = FlowerDataset('train', datasets_dir, transformer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = FlowerDataset('test', datasets_dir,transformer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

model = swin_tiny_patch4_window7_224(num_classes=5).to(device)

# 加载预训练模型
# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
weights_dict = torch.load('../swin_tiny_patch4_window7_224.pth', map_location=device)["model"]
# 删除有关分类类别的权重
for k in list(weights_dict.keys()):
    if "head" in k:
        del weights_dict[k]
model.load_state_dict(weights_dict, strict=False)
print("load previous weight!")


# def create_optimizer(lr):
#     return torch.optim.AdamW(model.parameters(), lr=lr)  #, weight_decay=5E-2

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=5E-2)

criterion = torch.nn.CrossEntropyLoss()

def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels.reshape(-1)
        inputs, labels = inputs.to(device), labels.to(device)

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index%50 == 49:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 50))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            labels = labels.reshape(-1)

            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            _, predicted = torch.max(output.data, dim = 1)
            # print("\npred:")
            # print(predicted)
            # print("\nlabel:")
            # print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %%\n' % (100 * correct / total))
    creat_file.write('accuracy on test set: %d %%\n' % (100 * correct / total) + "\n")
    return correct / total

if __name__ == "__main__":
    epoch_list = []
    acc_list = []
    prev_acc = 0

    creat_file.write("use GPU " + str(device) + " train!\n")

    print("use GPU " + str(device) + " train!")

    for epoch in range(15):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

        if acc > prev_acc:
            prev_acc = acc
            print("save acc equal to "+str(acc)+" 's model\n")
            creat_file.write("save acc equal to "+str(acc)+" 's model\n\n")
            torch.save(model.state_dict(), 'model/'+'swin_flower_acc_'+str(acc)+'.pth')

    # creat_file = open('log.txt','a+')
    # creat_file.write('lr='+str('%.10f'%lr)+'   no weight_decay'+'\n')
    # creat_file.write(str(epoch_list)+'\n'+str(acc_list)+'\n'+'\n')
    print(str(epoch_list)+'\n'+str(acc_list)+'\n'+'\n')
    creat_file.write(str(epoch_list)+'\n'+str(acc_list)+'\n'+'\n\n')