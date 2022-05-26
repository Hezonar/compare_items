# compare_items

To download a ready model load this file - https://drive.google.com/drive/folders/129NieUu-vXop-KHKdkSVh3bB7MnW_yOg?usp=sharing

# How to use ? 
You need to load model or train, after that you can run main.py and drop your json to neural_input_dir 

# Your json struct 

{\n
	\n"original_goods": {
	\n	"name": "... (text)",
	\n	"image": "... (from url or path ex. 123.jpg)",
	\n	"description": "... (text)",
	\n	"price": "... (float)"
	\n},
	\n"other_goods": \[
	\n	{
	\n		"ads_id": "... (ID of the compared items)",
	\n		"name": "... (text)",
	\n		"image": "....(from url or path ex. 123.jpg)",
	\n		"price": "... (float)",
	\n		"description": "...(text)"
	\n	},
    \n...
 \n ]
\n}

