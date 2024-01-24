import os
import numpy as np 
import matplotlib.pyplot as plt
import mlflow
import torch
import lightning as L
import multiprocessing as mp
import argparse
import shutil
import pickle

# Custom imports
from GAN_models import Simple_GAN
from unet_models import EncoderDecoder, UNet
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
    
    scaler = TransformStandardScaler()
    
    HPARAMS = {
        'data_dir': data_dir,
        'batch_size': 16,
        'lr': 1e-3,
        'generator_lr': 1e-3,
        'discriminator_lr': 1e-3,
        'epochs': 1,
        'num_workers': 4,
        'overfit_single_batch': True,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'pretransform': scaler,
        'transform': None,
        'load_into_memory': False,
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
    
    with mlflow.start_run(run_name=HPARAMS['model_name']):
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
            pretransform=None,
            batch_size=HPARAMS['batch_size'],
            num_workers=HPARAMS['num_workers'],
            transform=HPARAMS['transform'],
            device=HPARAMS['device'],
            load_into_memory=HPARAMS['load_into_memory']
        )
        dataModule.setup(stage='fit')
        
        fit_data = np.array([dataModule.train_dataset[i][0].detach().cpu().numpy() for i in range(len(dataModule.train_dataset))]).reshape(-1, 256, 256, 3)
        print(f'Fit data shape: {fit_data.shape}')
        scaler.fit(fit_data)
        print(f'Scaler fitted to data with shape {fit_data.shape}')
        print(f'Scaler mean: {scaler.mean} and std: {scaler.std}')
        
        # Set the pretransform to the scaler
        dataModule = CustomDataModule(
        data_dir=os.path.abspath(os.path.join(HPARAMS['data_dir'], 'splits')),
        pretransform=scaler,
        batch_size=HPARAMS['batch_size'],
        num_workers=HPARAMS['num_workers'],
        transform=HPARAMS['transform'],
        device=HPARAMS['device'],
        load_into_memory=HPARAMS['load_into_memory']
        )
        dataModule.setup(stage='train')
        
        train_dataloader = dataModule.train_dataloader()
        val_dataloader = dataModule.val_dataloader()
        print('Dataloaders created')
        
        example_batch = next(iter(train_dataloader))
        print('Example batch shape:', example_batch[0].shape)
        print('Example batch device:', example_batch[0].device)
        print('Values in example batch:', example_batch[0].detach().cpu().numpy().min(), example_batch[0].detach().cpu().numpy().max())
        
        example2_batch = train_dataloader.dataset.__getitems__([1, 2])
        print('Example batch shape:', example2_batch[0].shape)
        print('Example batch device:', example2_batch[0].device)
        print('Values in example batch:', example2_batch[0].detach().cpu().numpy().min(), example2_batch[0].detach().cpu().numpy().max())
        
        # Scale two images to [0, 1]
        example2 = np.array([example2_batch[0][0].detach().cpu().numpy(), example2_batch[0][1].detach().cpu().numpy()])
        example2 = example2.transpose(0, 2, 3, 1)
        example2 = scaler.inverse_transform(example2) / 255
        
        # Plot two images
        fig, ax = plt.subplots(2, 3)
        ax = ax.flatten()
        ax[0].imshow(example2[0])
        ax[1].imshow(example2_batch[1][0].detach().cpu().numpy().transpose(1, 2, 0))
        ax[2].imshow(example2[0] * example2_batch[1][0].detach().cpu().numpy().transpose(1, 2, 0))
        
        ax[3].imshow(example2[1])
        ax[4].imshow(example2_batch[1][1].detach().cpu().numpy().transpose(1, 2, 0))
        ax[5].imshow(example2[1] * example2_batch[1][1].detach().cpu().numpy().transpose(1, 2, 0))
        
        mlflow.log_figure(fig, 'input_example.png')
        plt.savefig(os.path.join(figure_dir, 'input_example.png'))
        
        print(f'Scaler mean: {scaler.mean} and std: {scaler.std}')
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
        elif HPARAMS['model_name'] == 'UNet':
            model = UNet(HPARAMS)
        else:
            raise ValueError(f'Unknown model name: {HPARAMS["model_name"]}')
        
        # Log hyperparameters
        mlflow.log_params(HPARAMS)
        
        # Make sure model is on the right device
        # model = model.to(HPARAMS['device'])
    
        # Set up trainer
        trainer = L.Trainer(
            # accelerator=HPARAMS['device'],
            # devices=1 if HPARAMS['device'] == 'cuda' else 'auto',
            max_epochs=HPARAMS['epochs'],
            callbacks=callbacks,
            # deterministic=False,
            overfit_batches=1 if HPARAMS['overfit_single_batch'] else 0,
            log_every_n_steps=1,
            # fast_dev_run=True,
            # precision=16 if HPARAMS['device'] == 'cuda' else 32,
            profiler='simple',
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
        
        # Pickle pretransform object
        pckl_path = os.path.join(model_dir, 'pretransform.pkl')
        with open(pckl_path, 'wb') as f:
            pickle.dump(HPARAMS['pretransform'], f)
            mlflow.log_artifact(pckl_path)
        
        # --- EVALUATE --- #
        if args.run_eval:
            dataModule.setup(stage='test')
            test_dataloader = dataModule.test_dataloader()
            
            model.eval()
            model.freeze()
            
            # Evaluate
            trainer.test(
                model=model,
                dataloaders=test_dataloader,
                ckpt_path='best'
            )
            
            # Get a batch of images
            batch = test_dataloader.dataset.__getitems__([0,1,2,3,4,5,6,7,8,9])
            
            # Make sure batch is on the same device as the model
            # batch = [b.cpu() for b in batch]            
            
            # Actually, just get 10 images
            # Unpack batch
            x, mask = batch[0, :], batch[1, :]
            
            # Generate inpainted image
            inpainted_images = model(x, mask)
            
            # Save the inpainted images
            for i, inpainted_image in enumerate(inpainted_images):
                original_image = x[i]
                original_mask = mask[i]
                original_image_masked = original_image * original_mask
                
                imgs = [original_image, original_mask, original_image_masked, inpainted_image]
                
                # Scale the images to [0, 1]
                imgs = np.array([img.detach().cpu().numpy() for img in imgs]).transpose(0, 2, 3, 1)
                
                # Inverse transform
                imgs = scaler.inverse_transform(imgs) / 255
                
                # Plot the images
                fig, ax = plt.subplots(1, 4)
                ax = ax.flatten()
                ax[0].imshow(imgs[0])
                ax[1].imshow(imgs[1])
                ax[2].imshow(imgs[2])
                ax[3].imshow(imgs[3])
                
                # Add labels
                ax[0].set_title('Original')
                ax[1].set_title('Mask')
                ax[2].set_title('Masked')
                ax[3].set_title('Inpainted')
                
                plt.tight_layout()
                
                # Log the figure
                mlflow.log_figure(fig, f'example_{i}.png')
                plt.savefig(os.path.join(figure_dir, f'example_{i}.png'))
