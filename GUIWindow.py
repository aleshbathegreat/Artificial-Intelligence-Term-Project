from FeatureExtraction import load_audio, mfccmethod, chromamethod, chroma_cqt, chroma_mfcc
import librosa
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg 
import EmotionLogic
import NeuralOptimised

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow, 
    QPushButton,
    QVBoxLayout,
    QHBoxLayout, 
    QWidget, 
    QMenu,
    QFileDialog,
    QToolButton, QLabel
)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.y = None
        self.sr = None
        self.audiofile = None
        self.setWindowTitle("Emote")
        self.setStyleSheet("background-color: #EEB6C7;")
        self.features = None
        self.valence = None
        self.arousal = None
        self.M = None

        #media player for audio/output
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)

        #button to pick an audio
        self.pick_audio_button = QPushButton()
        self.pick_audio_button.setText("Pick Audio")
        self.pick_audio_button.setFixedSize(QSize(75,20))
        self.pick_audio_button.clicked.connect(self.openaudio)

        #button to play audio
        self.play_audio_button = QPushButton()
        self.play_audio_button.setIcon(QIcon("icons/speaker.svg"))
        self.play_audio_button.setIconSize(QSize(60,60))
        self.play_audio_button.setFixedSize(QSize(68,68))
        self.play_audio_button.setEnabled(False)
        self.play_audio_button.clicked.connect(self.playaudio)

        #button to mute audio
        self.pause_audio_button = QPushButton()
        self.pause_audio_button.setIcon(QIcon("icons/speaker-mute.svg"))
        self.pause_audio_button.setIconSize(QSize(60,60))
        self.pause_audio_button.setFixedSize(QSize(68,68))
        self.pause_audio_button.setEnabled(False)
        self.pause_audio_button.clicked.connect(self.player.pause)

        self.pick_method_button = QToolButton()
        self.pick_method_button.setText("Select Feature Extraction")
        self.pick_method_menu = QMenu(self.pick_method_button)
        self.pick_method_menu.addAction("MFCC", lambda: self.select(1))
        self.pick_method_menu.addAction("Chroma", lambda:self.select(2))
        self.pick_method_menu.addAction("Chroma-MFCC", lambda:self.select(3))
        self.pick_method_button.setPopupMode(QToolButton.InstantPopup)
        self.pick_method_button.setMenu(self.pick_method_menu)
        
        #for optimizer
        self.pick_optimizer_button = QToolButton()
        self.pick_optimizer_button.setText("Select Optimiser")
        self.pick_optimizer_menu= QMenu(self.pick_optimizer_button)
        self.pick_optimizer_menu.addAction("Gradient Descent", lambda: self.selectoptimiser(1, self.M))
        self.pick_optimizer_menu.addAction("Simulated-Annealing/Descent", lambda:self.selectoptimiser(2, self.M))
        self.pick_optimizer_menu.addAction("Evolutionary Algorithm", lambda:self.selectoptimiser(3, self.M))
        self.pick_optimizer_button.setPopupMode(QToolButton.InstantPopup)
        self.pick_optimizer_button.setMenu(self.pick_optimizer_menu)

        #emogenerator
        self.generate_emo_button = QToolButton()
        self.generate_emo_button.setText("Generate Emotion")
        self.generate_emo_button.clicked.connect(lambda: self.emogen(self.valence, self.arousal))

        #improve emo label
        self.emolabel = QLabel("Calculating..")
        self.emolabel.setAlignment(Qt.AlignCenter)
        self.emolabel.setStyleSheet("""QLabel {
        font-size: 22px;
        font-weight: bold;
        color: #1f618d;}""") #highlighting emotion
        #reset button
        self.resetbutton = QPushButton()
        self.resetbutton.setText("Reset")
        self.resetbutton.clicked.connect(lambda: self.resetfunc())


        #graph placeholder
        self.audiovisualiser = QLabel()
        self.audiovisualiser.setText("Audio Visualiser")
        self.audiovisualiser.setAlignment(Qt.AlignCenter)
        self.audiovisualiser.setFixedSize(600, 500)
        self.audiovisualiser.setStyleSheet(""" QLabel{border: 3px dashed gray;
                                           background-color: #f8f8f8;
                                           color: #888}""")


        #FIX FORMAT
        container = QWidget()

        self.layout = QVBoxLayout()

        row = QVBoxLayout()
        row.addWidget(self.pick_audio_button)
        row.addSpacing(15)
        play_mutebox = QHBoxLayout()
        play_mutebox.addWidget(self.play_audio_button)
        play_mutebox.addWidget(self.pause_audio_button)
        play_mutebox.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        row.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        #row.setContentsMargins(0,50,0,0)

        row2 = QHBoxLayout()
        row2.addWidget(self.pick_method_button, alignment= Qt.AlignLeft)
        row2.addWidget(self.pick_optimizer_button, alignment = Qt.AlignRight)
        #row2.setContentsMargins(20, 10, 0, 0)
        

        row3 = QVBoxLayout()
        row3.addWidget(self.generate_emo_button)
        row3.addWidget(self.emolabel)
        row3.setAlignment(Qt.AlignBottom | Qt.AlignCenter)

        self.layout.addLayout(row)
        self.layout.setSpacing(10)
        self.layout.addLayout(play_mutebox)
        self.layout.addLayout(row2)
        self.layout.addWidget(self.audiovisualiser)
        self.layout.addLayout(row3)
        self.layout.addWidget(self.resetbutton)
        container.setLayout(self.layout)
        self.setCentralWidget(container)
        
    #all linked/selecter functions
    def openaudio(self):
        file, _ = QFileDialog.getOpenFileName(self,"Select Audio File", "", "Audio File (*.mp3 *.wav);;All Files(*)")
        
        if file:
            self.y, self.sr = load_audio(file)
            self.audiofile = file
            self.play_audio_button.setEnabled(True)
            self.pause_audio_button.setEnabled(True)
        return

    def playaudio(self):
        if self.audiofile:
            self.player.setSource(QUrl.fromLocalFile(self.audiofile))
            self.audio.setVolume(0.5)
            self.player.play()
        return
        
    def select(self, M):
        figure = Figure(figsize= (12,10), constrained_layout= True)
        self.visual = FigureCanvasQTAgg(figure)
        axes = figure.add_subplot(111) #one plot
        match M:
            case 1:
                self.M = 1
                self.features, mfcc, hop = mfccmethod(self.y, self.sr)
                bar = librosa.display.specshow(mfcc, x_axis= 'time', y_axis= 'mel', sr=self.sr, hop_length=hop, cmap = 'plasma', ax = axes)
                figure.colorbar(bar, ax = axes)
            case 2:
                self.M = 2
                self.features,chroma, hop= chromamethod(self.y, self.sr)
                bar = librosa.display.specshow(chroma, x_axis= 'time', y_axis= 'chroma', sr=self.sr, hop_length=hop, cmap = 'plasma', ax = axes)
                figure.colorbar(bar, ax = axes, format="%.2f")
            case 3:
                self.M = 3
                axis1 = figure.add_subplot(2,1,1)
                self.features, chroma, mfcc, hop = chroma_mfcc(self.y, self.sr)
                bar1 = librosa.display.specshow(mfcc, x_axis= 'time', y_axis= 'mel', sr=self.sr, hop_length=hop, cmap = 'plasma', ax = axis1)
                axis1.set(title = 'MFCC')
                figure.colorbar(bar1, ax = axis1, format="%.2f")

                axis2 = figure.add_subplot(2,1,2)
                bar2 = librosa.display.specshow(chroma, x_axis= 'time', y_axis= 'chroma', sr=self.sr, hop_length=hop, cmap = 'coolwarm', ax = axis2)
                axis2.set(title = 'Chroma')
                figure.colorbar(bar2, ax = axis2, format="%.2f")

        self.visual.draw()

        self.layout.replaceWidget(self.audiovisualiser, self.visual)
        self.audiovisualiser.deleteLater()
        self.audiovisualiser = None
        #self.layout.addWidget(visual)
        return
        
    def selectoptimiser(self, N, M):
        match N:
            case 1:
                match M:
                    case 1:
                        self.valence, self.arousal = NeuralOptimised.createmodel(self.features,'modelweightsmfccgradient.pth')
                    case 2:
                        self.valence, self.arousal = NeuralOptimised.createmodel(self.features,'model_weights_chromagradient.pth')
                    case 3:
                        self.valence, self.arousal = NeuralOptimised.createmodel(self.features,'model_weights_mfccchromagradient.pth')
            case 2:
                match M:
                    case 1:
                        self.valence, self.arousal = NeuralOptimised.createmodel(self.features,'model_weights_SA-GD-mfcc.pth')
                    case 2:
                        self.valence, self.arousal = NeuralOptimised.createmodel(self.features,'model_weights_SA-GD-chroma.pth')
                    case 3:
                        self.valence, self.arousal = NeuralOptimised.createmodel(self.features,'model_weights_SA-GDmfccchroma.pth')
            case 3:
                match M:
                    case 1:
                        self.valence, self.arousal = NeuralOptimised.createmodel(self.features,'emotion_model_weights_EA_GD_MFCC.pth')
                    case 2:
                        self.valence, self.arousal = NeuralOptimised.createmodel(self.features,'emotion_model_weights_EA (2).pth')
                    case 3:
                        self.valence, self.arousal = NeuralOptimised.createmodel(self.features,'emotion_model_weights_EA-GD.pth')
        return

    def emogen(self,val, ars):
        emote = EmotionLogic.EmotionLogic()
        emotion = emote.fuzzy(val,ars)
        self.emolabel.setText(f"Emotion:{emotion}")
        return

    def resetfunc(self):
        self.y = None
        self.sr = None
        self.audiofile = None
        self.features = None
        self.valence = None
        self.arousal = None
        self.M = None

        self.audiovisualiser = QLabel()
        self.audiovisualiser.setText("Audio Visualiser")
        self.audiovisualiser.setAlignment(Qt.AlignCenter)
        self.audiovisualiser.setFixedSize(600, 500)
        self.audiovisualiser.setStyleSheet(""" QLabel{
            border: 3px dashed gray;
            background-color: #f8f8f8;
            color: #888}""")

        self.layout.insertWidget(3, self.audiovisualiser)
        self.pause_audio_button.setEnabled(False)
        self.play_audio_button.setEnabled(False) 
        self.emolabel.setText("Calculating..")

if __name__ == "__main__":
    musicapp = QApplication([])
    window = Window()
    window.resize(400,500)
    window.show()
    musicapp.exec()
