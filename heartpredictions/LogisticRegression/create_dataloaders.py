import torch

def create_dataloaders(dataset, split_proportions, batch_size, display_informations=False):
    """
    Create three dataloaders (test, train, validation).

    Parameters:
        dataset (array) : The dataset array.
        split_proportions (list) : Proportion of each dataloaders (sum must be equal to 1).
        batch_size (int) : Batch size for the training and the validation dataloaders.
        display_informations (boolean) : Dislay lengths of dataloaders if True.

    Returns:
        train_dataloader (DataLoader), val_dataloader (DataLoader), test_dataloader (DataLoader)
    """
    lengths = [round(len(dataset) * split) for split in split_proportions]

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=False,
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True
    )

    if display_informations:
        print(f'Total dataset: {len(train_dataloader) + len(val_dataloader) + len(test_dataloader)}, '
            f'train dataset: {len(train_dataloader)}, val dataset: {len(val_dataloader)}, test_dataset: {len(test_dataloader)}')
    return train_dataloader, val_dataloader, test_dataloader

