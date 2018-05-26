# encoding:utf-8
from chapter11.preprocess import get_data, pre_process_data
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validatin'], loc='upper left')
    plt.show()


train_df, test_df = get_data()
train_features, train_label = pre_process_data(train_df)
test_features, test_label = pre_process_data(test_df)

# 定义模型
model = Sequential()
model.add(Dense(40, input_dim=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
print(model.summary())

# 定义训练方式
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 开始训练
train_history = model.fit(train_features, train_label, validation_split=0.1, epochs=30, batch_size=32)
# 绘制训练过程中准确率图
show_train_history(train_history, 'acc', 'val_acc')

# 保存结果
model.save('model_save.h5', overwrite=True)
# 测试数据评估
scores = model.evaluate(test_features, test_label)
print(scores[1])
