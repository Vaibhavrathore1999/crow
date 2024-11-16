import os

# Define the base directory containing the PACS dataset
base_dir = "../PACS"  # Replace with the path to your dataset
domains = ["art_painting", "cartoon", "photo", "sketch"]  # List of domains

def create_txt_files(base_dir, domains):
    for domain in domains:
        domain_path = os.path.join(base_dir, domain)
        if not os.path.exists(domain_path):
            print(f"Domain path not found: {domain_path}")
            continue
        
        # Collect all class directories
        classes = sorted(os.listdir(domain_path))  # Sort classes alphabetically
        class_to_label = {cls: idx for idx, cls in enumerate(classes)}  # Assign labels to classes
        
        # Prepare the .txt file
        output_file = f"/raid/biplab/sarthak/crow/Datasets/PACS/{domain}.txt"
        with open(output_file, "w") as txt_file:
            for cls in classes:
                class_path = os.path.join(domain_path, cls)
                if not os.path.isdir(class_path):
                    continue
                
                # List all images in the class directory
                images = sorted(os.listdir(class_path))  # Sort images alphabetically
                for img in images:
                    img_path = os.path.join(class_path, img)
                    if os.path.isfile(img_path):
                        # Write the image path and label to the .txt file
                        txt_file.write(f"{img_path} {class_to_label[cls]}\n")
        
        print(f"File created: {output_file}")

# Run the script
create_txt_files(base_dir, domains)
