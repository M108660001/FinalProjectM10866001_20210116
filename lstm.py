""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import os
os.walk('')
pitchRange =[] 
pitchRangeCount=[]
PitchMC = []

from os import walk

from keras.callbacks import EarlyStopping 
import keras.callbacks

detail=input('是否過程停頓檢查?(Y/N)')
epochs_times=int(input('epochs幾次?'))
batch_size_=int(input('batch_size多少?(原本32)'))

def train_network():
    """ 訓練神經網路生成音樂 """
    AllMC = openData()     #讀取midi_songs目錄下每一首曲子的檔名
    countP(AllMC)  #計算midi_songs目錄下所有曲子有哪些旋律跟次數
    print("選擇製作的模型使用的旋律 例如: 可以選擇G產生模型 或者選擇 G 跟Bb各產生一個模型 或者全部旋律組合產生一個模型")
    Pitchs=input("Choose Pitch?(ex: G  or G,Bb.. or All 之類)")
    if(Pitchs=="All"):               #判斷輸入字是否為All
        Pitchs = ""
        temp = 0
        for i in pitchRange:         #pitchRange 即所有你的資料有哪些旋律種類 如G大調 Ab調等
            Pitchs +=i
            if temp<len(pitchRange)-1:     #對所有的音樂旋律種類疊加組成你選擇的旋律
               Pitchs +=","
            temp+=1
        print(Pitchs)
       # Pitchs = Pitchs[-(len(Pitchs)-1)]
       # print(Pitchs)
    print("確認選擇使用:"+str(Pitchs)+"產生模型?")    #使用者確認是否使用這些旋律種類生成模型
    print("輸入Y 代表確定 輸入N代表取消  輸入All代表將上述你選擇的旋律音樂全部組合成一個模型")
    x=input("Right?(Y/N/All)")
    OnlyBestResult=input('是否刪除最好成果以外的模型參數(Y/N)')
    if x=="Y":                                    #使用者選擇的每種旋律各產生一個模型
        Pitchs=Pitchs.split(",")
        for Pitch in Pitchs:
            print(Pitch)
            ChooseMC=findMC(AllMC,Pitch)
            print(ChooseMC)
            notes = get_notes(ChooseMC,Pitch)

       # 獲取 pitch names(音高) 數量
            n_vocab = len(set(notes))

            network_input, network_output = prepare_sequences(notes, n_vocab)

            model = create_network(network_input, n_vocab)

            train(model, network_input, network_output,Pitch,OnlyBestResult)
    elif x=="All":                                 #使用者選擇的每種旋律合併產生一個模型
        ChooseAll = []
        youChoosePitchs = "Mix"                 
     #  everything
        for i in Pitchs:
            ChooseMC=findMC(AllMC,i)
            ChooseAll=ChooseAll+ChooseMC
            if i ==",":
                youChoosePitchs=youChoosePitchs+"_"   #這種模型檔名會是使用者選擇的旋律組合名=>例如選Ab,C 則製作出的模型放在Ab_C_Pitch的目錄下
            else:
                youChoosePitchs=youChoosePitchs+i
            
    
        print(ChooseAll)
        notes = get_notes(ChooseAll,youChoosePitchs+"_Pitch")    #取得對應的notes並且儲存

        n_vocab = len(set(notes))

        network_input, network_output = prepare_sequences(notes, n_vocab)  

        model = create_network(network_input, n_vocab) #建立網路

        train(model, network_input, network_output,youChoosePitchs+"_Pitch",OnlyBestResult) #訓練模型並儲存權重在weight目錄下

def openData():
    notes = []
    AllMC = []
    for file in glob.glob("midi_songs/*"):
        print("1."+file)
        file=glob.glob(file+"/*")
        for files in file:
            print("2."+files)
            dataMC = glob.glob(files+"/*.mid")
            for filess in dataMC:
                AllMC.append(filess)
    return AllMC

def countP(AllMC):
    for file in AllMC:   #尋找目錄下所有的音樂檔名
       a = file.split("\\")
       # print(a[len(a)-1])
       a=a[len(a)-1].split("_")    #分割  檔名是  C_音樂名稱.midi  => 要取出C的部分(旋律)
       #strinG += a[0]
       temp = 0
       find=0
       for i in pitchRange:              #尋找C是不是在已知的旋律名單中 是的話在該位置旋律計數+1
           if i==a[0]:
               pitchRangeCount[temp]+=1
               find=1
           temp+=1
       if find==0:                          #如果現在找到的音樂旋律是未出現過的，新增一種旋律並且+1計數
           pitchRange.append(a[0])
           pitchRangeCount.append(1)
    print(pitchRange)
    print(pitchRangeCount)
 
def findMC(AllMC,Pitch):           #尋找對應旋律音樂的方法  Pitch即是使用者選擇的旋律
    
    temp = 0
    no = 0
    ChooseMC = []
    for i in pitchRange:
        if Pitch==i:
            no=temp
        temp+=1
          
    for file in AllMC:
       a = file.split("\\")
       # print(a[len(a)-1])
       a=a[len(a)-1].split("_")
       #strinG += a[0]
       temp = 0
       if Pitch==a[0]:                 #尋找每個音樂檔，滿足Pitch的旋律的音樂檔名則留下
           ChooseMC.append(file)
    return ChooseMC               #回傳選擇了哪些音樂的檔名
                
def get_notes(ChooseMC,Pitch):   #從選擇的音樂清單中，取得每首音樂的音符組合起來當notes資料
    
    """ 從midi資料夾中取得所有文件的音符 """
    notes = []
                    
    for file in ChooseMC:                  #讀取找到的音樂 然後作音符判斷來儲存到list資料中 當訓練資料
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None                 

        try: # 檔案中是否有音符
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    print(notes)
    if detail=="Y":
        input("stop")
 #   print(notes[0])
    if ( os.path.isdir('data/'+Pitch)):      #判斷data下有沒有相同旋律目錄，沒有的話新建目錄並且儲存
        print("Directory exists!")
    else:
        os.mkdir('data/'+Pitch)
        print("Directory Create")
        
    with open('data/'+Pitch+'/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
     #   print (network_input)
     #   input("stop")   #一個network_input元素有100個音色
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    print(network_input)
    if detail=="Y":
        input("stop")
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)
    print(network_input)
    if detail=="Y":
        input("stop")
    print(network_output)
    if detail=="Y":
        input("stop")
    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()                    #建立LSTM訓練模型的架構
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output,Pitch,OnlyBestResult):  #正是訓練模型 儲存權重在weights之下

    """ train the neural network """
    if ( os.path.isdir("weights/"+Pitch)):
        print("Directory exists!")
    else:
        os.mkdir("weights/"+Pitch)
        print("Directory Create")
        
    filepath = "weights/"+Pitch+"/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    early_stopping=keras.callbacks.EarlyStopping(  # new   https://www.twblogs.net/a/5b8b38c82b717718832e216b
    monitor='val_loss', 
    patience=5, 
    verbose=0, 
    mode='min'
    )    # new
    callbacks_list = [checkpoint]
    early_stoppingList=[early_stopping]
  #  model.fit(network_input, network_output, epochs=3, batch_size=32, callbacks=early_stoppingList)  # new
    model.fit(network_input, network_output, epochs=epochs_times, batch_size=batch_size_, callbacks=callbacks_list)#80,64 其他使用方式
    
    if OnlyBestResult=="Y":                       #僅保留最佳結果模型，參數設定為"Y"的時候則把最低loss以外的模型結果刪除
        files = glob.glob('weights/'+Pitch+'/*')
        minLoss = files[0].split("-")[len(files[0].split("-"))-2]
        print("minLoss First:"+str(minLoss))
        if detail=="Y":
            input("stop")
        xxxx =0
        while xxxx <2:
            for file in files:
                print(file)
            
                fileX = file.split("-")
                print('minLoss is'+str(minLoss))
                print('now fileX is'+str(fileX[len(fileX)-2]))
                if fileX[len(fileX)-2]<= minLoss:
                    minLoss=fileX[len(fileX)-2]
                    print('minLoss change')
                
                
                else:
                    if xxxx==1:
                        os.remove(file)
                    print('kill data ')
                if detail=="Y":
                    input("stop")
            xxxx+=1
    
#epochs = 200
if __name__ == '__main__':
    train_network()
