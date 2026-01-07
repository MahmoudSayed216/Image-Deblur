MODEL_NAME =  "Pix2VoxSharp Pure"
OPTIM_TYPE =  'AdamW'
LR = 0.001
PATIENCE = 5




EPOCHS = 100
BATCH_SIZE =  32
START_EPOCH = 0
SAVE_EVERY: 3
  
  
  
  continue_from_checkpoint: True
  checkpoint_id: 2025-03-26_06-16-57
  checkpoint_type: best
