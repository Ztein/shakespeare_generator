import tensorflow as tf
from data_loader import download_shakespeare
import os

class ShakespeareGeneratorTF:
    def __init__(self):
        self.text_vec_layer = None
        self.n_tokens = None
        self.shakespeare_model = None
        self.encoded = None
        
    def prepare_text(self, text):
        """Prepare and encode the text data"""
        print(f"First 80 characters: {text[:80]}")
        
        # Create and adapt text vectorization layer
        self.text_vec_layer = tf.keras.layers.TextVectorization(
            split="character",
            standardize="lower"
        )
        self.text_vec_layer.adapt([text])
        
        # Encode text
        self.encoded = self.text_vec_layer([text])[0]
        self.encoded -= 2  # drop tokens 0 (pad) and 1 (unknown)
        self.n_tokens = self.text_vec_layer.vocabulary_size() - 2
        
        dataset_size = len(self.encoded)
        print(f"Dataset size: {dataset_size} characters")
        print(f"Vocabulary size: {self.n_tokens} characters")
        print(f"Encoded first 80 characters: {self.encoded[:80]}")
        
    def create_datasets(self, sequence_length=100, batch_size=32):
        """Create training, validation and test datasets"""
        def to_dataset(sequence, length, shuffle=False, seed=None):
            ds = tf.data.Dataset.from_tensor_slices(sequence)
            ds = ds.window(length + 1, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
            if shuffle:
                ds = ds.shuffle(100_000, seed=seed)
            ds = ds.batch(batch_size)
            ds = ds.map(lambda window: (window[:, :-1], window[:, 1:]))
            ds = ds.repeat()
            return ds.prefetch(tf.data.AUTOTUNE)
        
        # Split data into train, validation, and test sets
        train_set = to_dataset(self.encoded[:1_000_000], sequence_length, shuffle=True, seed=42)
        valid_set = to_dataset(self.encoded[1_000_000:1_060_000], sequence_length)
        test_set = to_dataset(self.encoded[1_060_000:], sequence_length)
        
        return train_set, valid_set, test_set
    
    def build_model(self):
        """Build and compile the model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.n_tokens, output_dim=16),
            tf.keras.layers.GRU(128, return_sequences=True),
            tf.keras.layers.Dense(self.n_tokens, activation="softmax")
        ])
        
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="nadam",
            metrics=["accuracy"]
        )
        
        # Create the full shakespeare model with preprocessing
        self.shakespeare_model = tf.keras.Sequential([
            self.text_vec_layer,
            tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
            model
        ])
        
        return model
    
    def train(self, model, train_set, valid_set, epochs=10):
        """Train the model"""
        # Calculate steps per epoch
        batch_size = 32  # This should match the batch_size in create_datasets
        steps_per_epoch = 1_000_000 // (batch_size * 100)  # dataset_size // (batch_size * sequence_length)
        validation_steps = 60_000 // (batch_size * 100)  # validation_size // (batch_size * sequence_length)
        
        # Create checkpoints directory if it doesn't exist
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Setup model checkpoint with .keras extension
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "shakespeare_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        )
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
        
        try:
            history = model.fit(
                train_set,
                validation_data=valid_set,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=[model_ckpt, early_stopping]
            )
            return history
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def generate_text(self, prompt, n_chars=50, temperature=1):
        """Generate text from a prompt"""
        def next_char(text, temp=temperature):
            y_proba = self.shakespeare_model.predict([text])[0, -1:]
            rescaled_logits = tf.math.log(y_proba) / temp
            char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
            return self.text_vec_layer.get_vocabulary()[char_id + 2]
        
        generated_text = prompt
        for _ in range(n_chars):
            generated_text += next_char(generated_text)
        return generated_text

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Initialize generator
    generator = ShakespeareGeneratorTF()
    
    # Load and prepare text
    shakespeare_text = download_shakespeare()
    generator.prepare_text(shakespeare_text)
    
    # Create datasets
    train_set, valid_set, test_set = generator.create_datasets()
    
    # Build and train model
    model = generator.build_model()
    history = generator.train(model, train_set, valid_set)
    
    # Generate some text
    prompt = "To be or not to be"
    generated = generator.generate_text(prompt, temperature=0.01)
    print(f"\nPrompt: {prompt}")
    print(f"Generated text: {generated}")

if __name__ == "__main__":
    main()