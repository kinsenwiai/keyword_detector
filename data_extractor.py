import os
import glob
from tqdm import tqdm
from functools import partial
import PyPDF2
import pytesseract
from PIL import Image
import json
import sys
from tqdm import tqdm
import io
from pdf2image import convert_from_path, convert_from_bytes
from multiprocessing import Pool, cpu_count
def save_text_to_dict(pdf_path, text, output_dict):
    """
    Save extracted text to a dictionary and write to a JSON file.
    
    Args:
    - pdf_path (str): Path of the PDF file being processed.
    - text (dict): Text extracted from the PDF, with page numbers as keys and text as values.
    - output_dict (dict): Dictionary to store extracted text from multiple PDFs.
    """
    if text:
        for page_key, page_text in text.items():
            output_dict[page_key] = page_text
        pdf_path = pdf_path.replace('/', '_')
        json_file_path = os.path.join("/home/users/kinjals/work_dir/debug/", f"{pdf_path}.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(dict(output_dict), json_file)

def process_pdf(pdf_path, output_dict):
    """
    Process a PDF file to extract text from each page.
    Would first check with pypdf and if its an empty output then
    go with ocr recognition.

    Args:
    - pdf_path (str): Path of the PDF file being processed.
    - output_dict (dict): Dictionary to store extracted text from multiple PDFs.
    """
    pdf = PyPDF2.PdfReader(pdf_path)
    text_by_page = {}
    for page_num in tqdm(range(len(pdf.pages))):
        text = pdf.pages[page_num].extract_text()
        if text == "":
            dst_pdf = PyPDF2.PdfWriter()
            dst_pdf.add_page(pdf.pages[page_num])

            pdf_bytes = io.BytesIO()
            dst_pdf.write(pdf_bytes)
            pdf_bytes.seek(0)
            image = convert_from_bytes(pdf_bytes.read())
            for img in image:
                text += pytesseract.image_to_string(img, lang='eng+hin')
      
        text_by_page[page_num] = text
    save_text_to_dict(pdf_path, text_by_page, output_dict)
def process_files_in_folder(folder_path):
    """
    Process PDF files within a folder using multiprocessing.
    
    Args:
    - folder_path (str): Path of the folder containing PDF files.
    """
    output_dict = {}
    for foldername, subfolders, filenames in tqdm(os.walk(folder_path)):
        for filename in filenames:
            if filename.endswith(".pdf"):
                file_path = os.path.join(foldername, filename)
                process_pdf(file_path, output_dict)
                output_dict = {}

def main():
 
                
    """
    Main function to initiate multiprocessing for PDF processing.
    """
    folder_path = "/scratchj/ehealth/Education/raw_data_pendrives/drive1/"

    # Calculate the number of processes to use (up to the number of CPU cores)
    num_processes = 10 

    # Create a Pool of worker processes
    pool = Pool(processes=num_processes)

    # Process files in the folder using multiprocessing
    pool.map(process_files_in_folder, [folder_path])

    # Close the Pool to release resources
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()


