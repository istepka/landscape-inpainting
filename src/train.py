import os
import numpy as np 
import matplotlib.pyplot as plt
import mlflow
import torch
import lightning as L
import multiprocessing as mp
import argparse
import shutil

# Custom imports
from GAN_models import Simple_GAN
from unet_models import EncoderDecoder
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
        'batch_size': 128,
        'lr': 1e-3,
        'generator_lr': 1e-3,
        'discriminator_lr': 1e-3,
        'epochs': 10,
        'num_workers': mp.cpu_count(),
        'overfit_single_batch': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'pretransform': TransformStandardScaler(),
        'transform': None,
        'load_into_memory': True,
        'seed': seed,
        'optimizer': 'Adam',
        'weight_decay': 1e-5,
        'experiment_name': 'Simple_GAN',
        'model_name': 'Simple_GAN',
        'generate_data_even_if_already_exists': False,
        'generator_filters': [3, 32, 64, 32, 16, 8],
        'discriminator_filters': [3, 16, 32, 64],
        'pixels': 256,
    }
    
    #python src/train.py [OPTIONS OPTIONS: --run_eval TRUE, --device GPU, --experiment_name JUST_GAN]
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--run_eval', type=bool, default=True)
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--experiment_name', type=str, default='EncoderDecoder')
    argparser.add_argument('--model_name', type=str, default='EncoderDecoder')
    args = argparser.parse_args()
    # paste args into hparams
    for arg in vars(args):
        HPARAMS[arg] = getattr(args, arg)
    
    
    HPARAMS['pretransform'] = TransformStandardScaler()
    
    
    print('Creating splits')
    
    
    if os.path.exists(os.path.join(HPARAMS['data_dir'], 'splits')) \
        and os.path.exists(os.path.join(HPARAMS['data_dir'], 'splits', 'train')) \
        and os.path.exists(os.path.join(HPARAMS['data_dir'], 'splits', 'val')) \
        and os.path.exists(os.path.join(HPARAMS['data_dir'], 'splits', 'test')) \
        and not HPARAMS['generate_data_even_if_already_exists']:
        print('Splits already exist, no point in creating them again')
    else:
        print('Creating splits')
        shutil.rmtree(os.path.join(HPARAMS['data_dir'], 'splits'), ignore_errors=True)
        create_splits(
            data_dir=os.path.join(HPARAMS['data_dir'], 'processed'),
            target_dir=os.path.join(HPARAMS['data_dir'], 'splits'),
            dataset_size=2000, # Can be None to use all images
            train_prop=0.6,
            val_prop=0.2,
            test_prop=0.2
        )
    
    torch.cuda.empty_cache()
    
    print('Creating data module')
    dataModule = CustomDataModule(
        data_dir=os.path.abspath(os.path.join(HPARAMS['data_dir'], 'splits')),
        pretransform=HPARAMS['pretransform'],
        batch_size=HPARAMS['batch_size'],
        num_workers=1,
        transform=HPARAMS['transform'],
        device=HPARAMS['device'],
        load_into_memory=HPARAMS['load_into_memory']
    )
    dataModule.setup()
    
    train_dataloader = dataModule.train_dataloader()
    val_dataloader = dataModule.val_dataloader()
    print('Dataloaders created')
    
    example_batch = next(iter(train_dataloader))
    print('Example batch shape:', example_batch[0].shape)
    print('Example batch device:', example_batch[0].device)

    # Prepare stuff for training
    callbacks = [
        L.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss'),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        L.pytorch.callbacks.ModelSummary(max_depth=2)
    ]
    
    # Set up model
    if HPARAMS['model_name'] == 'Simple_GAN':
        model = Simple_GAN(HPARAMS)
    elif HPARAMS['model_name'] == 'EncoderDecoder':
        model = EncoderDecoder(HPARAMS)
    else:
        raise ValueError(f'Unknown model name: {HPARAMS["model_name"]}')
    
    
    
    with mlflow.start_run(run_name=HPARAMS['model_name']):
        
        # Log hyperparameters
        mlflow.log_params(HPARAMS)
    
        # Set up trainer
        trainer = L.Trainer(
            accelerator=HPARAMS['device'],
            devices=1 if HPARAMS['device'] == 'cuda' else 'auto',
            max_epochs=HPARAMS['epochs'],
            callbacks=callbacks,
            deterministic=True,
            overfit_batches=1 if HPARAMS['overfit_single_batch'] else 0,
            log_every_n_steps=1,
            # fast_dev_run=True,
            # precision=16 if HPARAMS['device'] == 'cuda' else 32,
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
                    
                # Log figure
                mlflow.log_figure(fig, f'inpainted_image_{i}.png')
                
                # Save figure
                plt.savefig(os.path.join(figure_dir, f'inpainted_image_{i}.png'))
                plt.close()
                
            

            
            
                
                
                
            
        
        
    
    