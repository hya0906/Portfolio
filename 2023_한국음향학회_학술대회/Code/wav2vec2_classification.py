import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFWav2Vec2Model
from tensorflow import keras
MODEL_CHECKPOINT = "facebook/wav2vec2-base"  # Name of pretrained model from Hugging Face Model Hub

# Set random seed
tf.keras.utils.set_random_seed(42)

# Maximum duration of the input audio file we feed to our Wav2Vec 2.0 model.
MAX_DURATION = 1
# Sampling rate is the number of samples of audio recorded every second
SAMPLING_RATE = 16000
BATCH_SIZE = 2 #32  # Batch-size for training and evaluating our model.
NUM_CLASSES = 8  # Number of classes our dataset will have (11 in our case).
HIDDEN_DIM = 768 #1024  # Dimension of our model output (768 in case of Wav2Vec 2.0 - Base).
MAX_SEQ_LENGTH = 80000  # Maximum length of the input audio file.
# Wav2Vec 2.0 results in an output frequency with a stride of about 20ms.
MAX_FRAMES = 249
MAX_EPOCHS = 10  # Maximum number of training epochs.
WAV_DATA_POINTS = 90000 #110000

def mean_pool(hidden_states, feature_lengths):
    attenion_mask = tf.sequence_mask(
        feature_lengths, maxlen=MAX_FRAMES, dtype=tf.dtypes.int64
    )
    padding_mask = tf.cast(
        tf.reverse(tf.cumsum(tf.reverse(attenion_mask, [-1]), -1), [-1]),
        dtype=tf.dtypes.bool,
    )
    hidden_states = tf.where(
        tf.broadcast_to(
            tf.expand_dims(~padding_mask, -1), (BATCH_SIZE, MAX_FRAMES, HIDDEN_DIM)
        ),
        0.0,
        hidden_states,
    )
    pooled_state = tf.math.reduce_sum(hidden_states, axis=1) / tf.reshape(
        tf.math.reduce_sum(tf.cast(padding_mask, dtype=tf.dtypes.float32), axis=1),
        [-1, 1],
    )
    return pooled_state


class TFWav2Vec2ForAudioClassification(layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, model_checkpoint, num_classes):
        super().__init__()
        # Instantiate the Wav2Vec 2.0 model without the Classification-Head
        self.wav2vec2 = TFWav2Vec2Model.from_pretrained(
            model_checkpoint, apply_spec_augment=False, from_pt=True
        )
        #self.wav2vec2.wav2vec2.feature_extractor._freeze_parameters()
        #self.wav2vec2.feature_extractor._freeze_parameters()
        self.pooling = layers.GlobalAveragePooling1D()
        # Drop-out layer before the final Classification-Head
        self.intermediate_layer_dropout = layers.Dropout(0.1)
        self.middle_layer = layers.Dense(HIDDEN_DIM)
        # Classification-Head
        self.final_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        # We take only the first output in the returned dictionary corresponding to the
        # output of the last layer of Wav2vec 2.0
        hidden_states = self.wav2vec2(inputs["input_values"])[0]

        # # If attention mask does exist then mean-pool only un-masked output frames
        # if tf.is_tensor(inputs["attention_mask"]):
        #     # Get the length of each audio input by summing up the attention_mask
        #     # (attention_mask = (BATCH_SIZE x MAX_SEQ_LENGTH) ∈ {1,0})
        #     audio_lengths = tf.cumsum(inputs["attention_mask"], -1)[:, -1]
        #     # Get the number of Wav2Vec 2.0 output frames for each corresponding audio input
        #     # length
        #     feature_lengths = self.wav2vec2.wav2vec2._get_feat_extract_output_lengths(
        #         audio_lengths
        #     )
        #     pooled_state = mean_pool(hidden_states, feature_lengths)
        # # If attention mask does not exist then mean-pool only all output frames
        # else:
        #     pooled_state = self.pooling(hidden_states)

        pooled_state = self.pooling(hidden_states)
        # middle_state = self.middle_layer(pooled_state)
        # intermediate_state = self.intermediate_layer_dropout(middle_state)
        # final_state = self.final_layer(intermediate_state)

        return pooled_state #final_state



def build_model():
    # Model's input
    inputs = {
        "input_values": tf.keras.Input(shape=(MAX_SEQ_LENGTH,), dtype="float32"),
        "attention_mask": tf.keras.Input(shape=(MAX_SEQ_LENGTH,), dtype="int32"),
    }
    # Instantiate the Wav2Vec 2.0 model with Classification-Head using the desired
    # pre-trained checkpoint
    wav2vec2_model = TFWav2Vec2ForAudioClassification(MODEL_CHECKPOINT, NUM_CLASSES)(
        inputs
    )
    x = layers.Dropout(0.1)(wav2vec2_model)
    x = layers.Dense(HIDDEN_DIM)(x)
    # Classification-Head
    x = layers.Dense(7)(x) #activation="softmax"


    # Model
    model = tf.keras.Model(inputs = inputs, outputs = x)#wav2vec2_model)
    # Loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    # Compile and return
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model
