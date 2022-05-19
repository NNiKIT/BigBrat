import albumentations as A

transform = A.Compose(
    [   
        A.OneOf([A.Rotate(limit=45, p=0.5), A.HorizontalFlip(p=0.5),], p=1),
        A.OneOf(
            [
                A.RandomContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            ],
            p=0.7,
        ),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
