""" This module generates notes for a midi file using the
    trained neural network """
import glob
import pickle
import numpy
import music21
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
import random

def generate():
    """ Generate a piano midi file """
    PitchMix2 = input("產生音樂格式? (舉例: 若要產生500個G調模型產生的音符+100個C調模型產生的音符組成的樂譜，輸入 500_G,100_C)")
    PitchMix=PitchMix2.split(",")
    prediction_output = []  #預測的音符
    for ii in PitchMix:
        PitchLong=ii.split("_")
        notes_Quantity=int(PitchLong[0])   #取得要產生幾個音符
        Pitch=PitchLong[1]                 #取得要使用哪個旋律的模型
    #load the notes used to train the model
   # Pitch = input("Choose Pitch?(ex: G  or Bb)")
    
        with open('data/'+Pitch+'/notes', 'rb') as filepath:
            notes = pickle.load(filepath)

    # Get all pitch names
        pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
        n_vocab = len(set(notes))

        network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
        model = create_network(normalized_input, n_vocab,Pitch)      
        prediction_output = generate_notes(model, network_input, pitchnames, n_vocab,prediction_output,notes_Quantity)
    create_midi(prediction_output,PitchMix2)  #生成音樂

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab,Pitch):
    """ create the structure of the neural network """
    model = Sequential()
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
    files =""
    i = 0
    print("Choose your weights:  ex 0 or 1 or 2")
    files = glob.glob('weights/'+Pitch+'/*')
    # Load the weights to each node              #讀取weights權重
    minLoss = files[0].split("-")[len(files[0].split("-"))-2]
    chooseWeight=""   #最後選擇哪個模型  =>loss小的
    for file in files:
        fileX = file.split("-")
        if fileX[len(fileX)-2]<= minLoss:
                    minLoss=fileX[len(fileX)-2]
                    print('minLoss change')
                    chooseWeight = file
       # print(str(i)+"."+file)
        i+=1
    
  #  x = int(input(""))
    print("you use :"+chooseWeight)
    
    model.load_weights(chooseWeight) # weights.hdf5

    return model

def generate_notes(model, network_input, pitchnames, n_vocab,prediction_output,notes_Quantity):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    

    # generate 500 notes
    for note_index in range(notes_Quantity):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        predictionX=prediction[0]
   #     print(predictionX)   #預測下一個音符所有可能的機率
        k=0
        maxP = predictionX[0]   # 第一名的機率
        prediction_Choose=0     # 第一名是第幾位
        for i in predictionX:
            if float(i)>float(maxP):
                maxP=predictionX[k]
                prediction_Choose=k
            k+=1
        k=0
        a=random.randint(0,len(predictionX)-1)
        max2P = predictionX[a]   # 第二名的機率
        prediction_Choose2=a     # 第二名是第幾位
        for i in predictionX:
            if float(i)>float(max2P):
                if k != prediction_Choose:
                    a=random.randint(1,4)     #隨機定義第二名
                    if a<4:
                        max2P=predictionX[k]
                        prediction_Choose2=k
            k+=1
         
     #   print(str(prediction_Choose))   #他預測下個音符是多少
     #   print(str(prediction_Choose2))   #他預測下個音符是多少(隨機的第二名)
        prediction2 = model.predict_classes(prediction_input, verbose=0)
     #   print(prediction2)    #他真實預測下個音符是多少
        index = numpy.argmax(prediction)
      #  print(index)
        a=random.randint(1,3)
        if a<2:
            index=prediction_Choose  #僅用第一名預測的音符寫譜
        else:
            index=prediction_Choose2   #用隨機的第二名寫譜
        result = int_to_note[index]
        
        
        prediction_output.append(result)
        

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output,Pitch):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='output\\'+Pitch+'_test_output.mid')   #在output目錄下生成mid檔

if __name__ == '__main__':
    generate()
