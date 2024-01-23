import os
import numpy as np 
import matplotlib.pyplot as plt
import mlflow
import torch
import lightning as L
import multiprocessing as mp
import argparse

# Custom imports
from transformations import TransformClass, TransformStandardScaler
from dataloaders import CustomDataset, CustomDataModule
from utils import create_splits

if __name__ == '__main__':
    
    # Seed for reproducibility
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed)

    # --- PREP ENVIRONMENT --- #

    # Set up all directories
    data_dir = 'data'
    model_dir = 'models'
    figure_dir = 'figures'
    mlflow_dir = 'mlflow'

    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(mlflow_dir, exist_ok=True)
    
    
    HPARAMS = {
        'data_dir': data_dir,
        'batch_size': 32,
        'lr': 1e-3,
        'epochs': 2,
        'num_workers': mp.cpu_count(),
        'overfit_single_batch': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'pretransform': TransformStandardScaler(),
        'transform': None,
        'load_into_memory': False,
        'seed': seed,
        'optimizer': 'Adam',
        'weight_decay': 1e-5,
        'experiment_name': 'Simple_GAN',
        'model_name': 'Simple_GAN',
    }
    
    #python src/train.py [OPTIONS OPTIONS: --run_eval TRUE, --device GPU, --experiment_name JUST_GAN]
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--run_eval', type=bool, default=False)
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--experiment_name', type=str, default='Simple_GAN')
    argparser.add_argument('--model_name', type=str, default='Simple_GAN')
    args = argparser.parse_args()
    # paste args into hparams
    for arg in vars(args):
        HPARAMS[arg] = getattr(args, arg)
    
    
    mlflow_logger = L.pytorch.loggers.MLFlowLogger(experiment_name=HPARAMS['experiment_name'], tracking_uri='file:' + os.path.abspath(mlflow_dir))
    
    HPARAMS['pretransform'] = TransformStandardScaler()
    
    
    print('Creating splits')
    create_splits(
        data_dir=HPARAMS['data_dir'],
        target_dir=os.path.join(HPARAMS['data_dir'], 'splits'),
        train_prop=0.8,
        val_prop=0.1,
        test_prop=0.1
    )
    
    
    print('Creating data module')
    dataModule = CustomDataModule(
        data_dir=HPARAMS['data_dir'],
        pretransform=HPARAMS['pretransform'],
        batch_size=HPARAMS['batch_size'],
        num_workers=HPARAMS['num_workers'],
        transform=HPARAMS['transform'],
        device=HPARAMS['device'],
        load_into_memory=HPARAMS['load_into_memory']
    ).to(HPARAMS['device'])
    
    train_dataloader = dataModule.train_dataloader().to(HPARAMS['device'])
    val_dataloader = dataModule.val_dataloader().to(HPARAMS['device'])
    
    # Prepare stuff for training
    callbacks = [
        L.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        L.callbacks.ModelCheckpoint(monitor='val_loss'),
    ]
    
    # Set up model
    if HPARAMS['model_name'] == 'Simple_GAN':
        model = L.Simple_GAN(HPARAMS).to(HPARAMS['device'])
    else:
        raise ValueError(f'Unknown model name: {HPARAMS["model_name"]}')
    model = model.to(HPARAMS['device'])
    
    # Log hparams
    mlflow_logger.experiment.log_params(HPARAMS)
    
    # Set up trainer
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=HPARAMS['epochs'],
        callbacks=callbacks,
        deterministic=True,
        overfit_batches=1 if HPARAMS['overfit_single_batch'] else 0,
        progress_bar_refresh_rate=1,
        checkpoint_callback=True,
        logger=mlflow_logger,
    )
    
    # Train
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    
    # Lightning save the model completely
    trainer.save_checkpoint(os.path.join(model_dir, 'model.ckpt'))
    
    
    # --- EVALUATE --- #
    if args.run_eval:
        test_dataloader = dataModule.test_dataloader().to(HPARAMS['device'])
        
        model.eval()
        model.freeze()
        
        # Evaluate
        trainer.test(
            model=model,
            test_dataloaders=test_dataloader
        )
        
        # Generate a couple of images
        model.eval()
        model.freeze()
        
        # Get a batch of images
        batch = next(iter(test_dataloader))
        
        # Unpack batch
        x, mask = batch
        
        # Generate inpainted image
        inpainted_images, scores = model(x, mask)
        
        # Save the inpainted images
        for i, inpainted_image in enumerate(inpainted_images):
            original_image = x[i]
            original_mask = mask[i]
            original_image_masked = original_image * original_mask
            
            imgs = [original_image, original_mask, original_image_masked, inpainted_image]
            # stack to tensor (N, C, W, H)
            imgs = torch.stack(imgs)
                        
            # use pretransform to unnormalize
            imgs = HPARAMS['pretransform'].inverse_transform(imgs)
            
            # unstack to list of tensors
            imgs = torch.unbind(imgs)
            
            # convert to numpy arrays
            imgs = [img.cpu().numpy() for img in imgs]
            
            fig, ax = plt.subplots(1, 4)
            
            # Plot each image along with label
            titles = ['Original Image', 'Mask', 'Masked Image', 'Inpainted Image']
            for j, img in enumerate(imgs):
                ax[j].imshow(img)
                ax[j].set_title(titles[j])
                ax[j].axis('off')
                
            # Save figure
            plt.savefig(os.path.join(figure_dir, f'inpainted_image_{i}.png'))
            plt.close()
            
        # Log images
        mlflow_logger.experiment.log_artifacts(figure_dir)
        

        
        
            
            
            
            
        
        
    
    