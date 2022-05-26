# compare_items

To download a ready model load this file - https://drive.google.com/drive/folders/129NieUu-vXop-KHKdkSVh3bB7MnW_yOg?usp=sharing

# How to use ? 
You need to load model or train, after that you can run main.py and drop your json to neural_input_dir 

# Your json struct 

{
	"original_goods": {
		"name": "... (text)",
		"image": "... (from url or path ex. www\.../.../123.jpg)",
		"description": "... (text)",
		"price": "... (float)"
	},
	"other_goods": \[
		{
			"ads_id": "... (ID of the compared items)",
			"name": "... (text)",
			"image": "....(from url or path ex. 123.jpg)",
			"price": "... (float)",
			"description": "...(text)"
		},
    ...
  ]
}

