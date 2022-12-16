from global_imports import *
from model import *
from model_proposed import *

parser = argparse.ArgumentParser(description='Train or Evaluate Cascaded Model')
parser.add_argument('-train', action='store_true')
parser.add_argument('-evaluate', action='store_true')
parser.add_argument('-model_type', required=True, choices=['cascaded', 'baseline'])

args = parser.parse_args()

imgs_dir='data/BraTS'
all_dirs = glob(f'{imgs_dir}/*')
all_dirs.sort()

# Shuffle Data
train_dirs, valid_dirs = shuffle_split(all_dirs, seed = 1)

# Data Augmentation
trn_tfms = A.Compose(
[
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RandomRotate90(p=0.3),
    A.ElasticTransform(alpha=10, alpha_affine=10),
    ToTensorV2(transpose_mask=True)
])
val_tfms = A.Compose([ToTensorV2(transpose_mask=True)])


train_ds = BratsDataset(train_dirs, modality_types, transform=trn_tfms)
valid_ds = BratsDataset(valid_dirs, modality_types, transform=val_tfms)
train_dl = DataLoader(train_ds, batch_size = 8, shuffle = True, num_workers = 2, pin_memory = True)
valid_dl = DataLoader(valid_ds, batch_size = 8, shuffle = True, num_workers = 2, pin_memory = True)


batch_size = 8
train_ds = BratsDataset(train_dirs, modality_types)
valid_ds = BratsDataset(valid_dirs, modality_types)
train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
valid_dl = DataLoader(valid_ds, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
print(f'Validation Set Size: {len(valid_dl)} | Training Set Size: {len(train_dl)}')

device = torch.device (DEVICE if torch.cuda.is_available() else 'cpu')
print(f'******* Device set {device} ****************')

if args.model_type == 'cascaded':
    Model = CascadedModel
    Trainer = CasacdedModelTrainer
    output_model_fn = 'best_model_cascaded_new.pth'
else:
    Model = UNet
    Trainer = BaselineModelTrainer
    output_model_fn = 'best_model_baseline.pth'

model = Model(n_channels=4, n_classes=4).to(device).float()
if args.train:
    trainer = Trainer(net=model,
                    train_dl=train_dl,
                    val_dl=valid_dl,
                    criterion=BCEDiceLoss(),
                    lr=5e-4,
                    accumulation_steps=batch_size,
                    batch_size=batch_size,
                    num_epochs=10,
                    output_model_fn=output_model_fn
                    )
    print(f'********** TRAINING MODEL *****************')
    trainer.run()
    print(f'********* TRAIN DONE **********************')

if args.evaluate:
    if not args.train:
        # Load saved model
        print(f'******** Evaluating saved {args.model_type} model on Validation Set *********')
        model.load_state_dict(torch.load(output_model_fn, map_location=DEVICE))
        model.to(DEVICE)
    model.eval()
    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(model, valid_dl, ['BG', 'TC', 'ED', 'ET'], 
                                                    cascaded_model=True if args.model_type=='cascaded' else False)

    dice_df = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns = ['BG dice', 'TC dice', 'ED dice', 'ET dice']
    iou_df = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns = ['BG jaccard', 'TC jaccard', 'ED jaccard', 'ET jaccard']
    val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metics_df = val_metics_df.loc[:, ['BG dice', 'BG jaccard', 
                                        'TC dice', 'TC jaccard', 
                                        'ED dice', 'ED jaccard',
                                        'ET dice', 'ET jaccard']]
    val_metics_df.sample(5)
    colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    palette = sns.color_palette(colors, 6)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=val_metics_df.mean().index, y=val_metics_df.mean(), palette=palette, ax=ax);
    ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15)
    ax.set_title(f"Eavluation metrics for {args.model_type} model", fontsize=20)

    for idx, p in enumerate(ax.patches):
            percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
            x = p.get_x() + p.get_width() / 2 - 0.15
            y = p.get_y() + p.get_height()
            ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    fig.savefig(f"plots/evaluation_bar_plot-{args.model_type}.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
    print(f'******* Evaluation metrics saved ********')

    # collect raw data
    val_metics_df.to_csv(f'metrics_{args.model_type}.csv')

