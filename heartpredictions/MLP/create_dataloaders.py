import torch

def create_dataloaders(dataset, split_proportions, batch_size, display_informations=False):
    lengths = [round(len(dataset) * split) for split in split_proportions]
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))
        
    train_class = []
    for _, (data, label) in enumerate(train_dataset):
        weight = 1 - dataset.class_weights[torch.argmax(label)]
        train_class.append(weight)
        
    sampler_train = torch.utils.data.WeightedRandomSampler(train_class, len(train_class))
    
    val_class = [1 - dataset.class_weights[torch.argmax(e[1])] for _,e in enumerate(val_dataset)]
    sampler_val = torch.utils.data.WeightedRandomSampler(val_class, len(val_class))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=1,
        persistent_workers=False,
        pin_memory=True,
        sampler=sampler_train
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True,
        sampler=sampler_val
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True,
        #sampler=sampler_test
    )

    if display_informations:
        print(f'Total dataset: {len(train_dataloader) + len(val_dataloader) + len(test_dataloader)}, '
            f'train dataset: {len(train_dataloader)}, val dataset: {len(val_dataloader)}, test_dataset: {len(test_dataloader)}')
    return train_dataloader, val_dataloader, test_dataloader

