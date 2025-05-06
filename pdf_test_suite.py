import os
import argparse
import subprocess
from PIL import Image # For pytesseract
import io # For pytesseract
import datetime # For comparison report timestamp

# Langchain (as you had it)
from langchain_community.document_loaders import PyPDFLoader

# PyMuPDF
import fitz  # PyMuPDF

# pdfminer.six
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams

# pytesseract
import pytesseract
# Configure Tesseract path if needed (uncomment and set if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Windows example
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract' # macOS/Linux example

# --- Configuration ---
DEFAULT_PDF_FILENAME = "my_document.pdf" # Make sure this file exists or provide one
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_OCR_PDF = os.path.join(SCRIPT_DIR, "temp_ocr_output.pdf")
TEMP_PAGE_IMAGE = os.path.join(SCRIPT_DIR, "temp_page_image.png")

# --- Extraction Functions for Different Methods ---

def extract_with_pypdfloader(pdf_path):
    """Langchain's PyPDFLoader (uses pypdf)."""
    print(f"\n--- Method: Langchain PyPDFLoader (pypdf) ---")
    if not os.path.exists(pdf_path):
        # This check is now redundant as main() checks first, but good for standalone use
        print(f"Error: PDF file not found at: {pdf_path}")
        return None
    loader = PyPDFLoader(pdf_path)
    try:
        pages_data = loader.load()
    except Exception as e:
        print(f"Error loading PDF with PyPDFLoader: {e}")
        return None
    if not pages_data: return None
    return [doc.page_content for doc in pages_data]

def extract_with_pymupdf(pdf_path):
    """PyMuPDF (fitz)."""
    print(f"\n--- Method: PyMuPDF (fitz) ---")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at: {pdf_path}")
        return None
    extracted_pages_text = []
    doc = None
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_pages_text.append(page.get_text("text")) # "text", "html", "xml", "blocks", "words"
    except Exception as e:
        print(f"Error with PyMuPDF: {e}")
        return None
    finally:
        if doc:
            doc.close()
    return extracted_pages_text

def extract_with_pdfminer(pdf_path):
    """pdfminer.six."""
    print(f"\n--- Method: pdfminer.six ---")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at: {pdf_path}")
        return None
    try:
        all_text = pdfminer_extract_text(pdf_path, laparams=LAParams())
        return [all_text] if all_text else None
    except Exception as e:
        print(f"Error with pdfminer.six: {e}")
        return None

def extract_with_ocrmypdf(pdf_path):
    """OCRmyPDF (creates a new OCR'd PDF, then extracts from it using PyMuPDF)."""
    print(f"\n--- Method: OCRmyPDF (then PyMuPDF) ---")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at: {pdf_path}")
        return None

    if os.path.exists(TEMP_OCR_PDF):
        try:
            os.remove(TEMP_OCR_PDF) # Clean up previous run
        except OSError as e:
            print(f"Warning: Could not remove existing TEMP_OCR_PDF: {e}")


    try:
        print(f"Running OCRmyPDF on {os.path.basename(pdf_path)} (this may take a while)...")
        # cmd = ["ocrmypdf", pdf_path, TEMP_OCR_PDF, "--skip-text"]
        cmd = ["ocrmypdf", pdf_path, TEMP_OCR_PDF, "--force-ocr"] # More likely to produce comparable OCR output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        print("OCRmyPDF completed.")
        if result.stderr:
             print("OCRmyPDF stderr:", result.stderr)

        if os.path.exists(TEMP_OCR_PDF):
            # Create a new PyMuPDF instance for extracting from the OCR'd file
            # This is important because extract_with_pymupdf prints its own method header
            print(f"--- Extracting text from OCR'd PDF using PyMuPDF ---")
            ocr_doc = None
            extracted_ocr_pages_text = []
            try:
                ocr_doc = fitz.open(TEMP_OCR_PDF)
                for page_num in range(len(ocr_doc)):
                    page = ocr_doc.load_page(page_num)
                    extracted_ocr_pages_text.append(page.get_text("text"))
                return extracted_ocr_pages_text
            except Exception as e_extract:
                print(f"Error extracting text from OCR'd PDF with PyMuPDF: {e_extract}")
                return None
            finally:
                if ocr_doc:
                    ocr_doc.close()
        else:
            print("OCRmyPDF did not produce an output file.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"OCRmyPDF failed: {e}")
        print(f"OCRmyPDF stdout: {e.stdout}")
        print(f"OCRmyPDF stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("OCRmyPDF command not found. Is it installed and in your system PATH?")
        return None
    # 'finally' for TEMP_OCR_PDF cleanup is handled in main() to allow inspection if needed

def extract_with_pytesseract(pdf_path, dpi=300):
    """PyTesseract (OCR page by page after converting to images with PyMuPDF)."""
    print(f"\n--- Method: PyTesseract (via PyMuPDF for images) ---")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at: {pdf_path}")
        return None

    extracted_pages_text = []
    doc = None
    try:
        doc = fitz.open(pdf_path)
        print(f"Found {len(doc)} pages. Processing with Tesseract (this may take time)...")
        for page_num in range(len(doc)):
            print(f"  OCR'ing page {page_num + 1}/{len(doc)}...")
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            try:
                page_text = pytesseract.image_to_string(img)
                extracted_pages_text.append(page_text)
            except pytesseract.TesseractNotFoundError:
                print("Tesseract command not found. Is it installed and in PATH, or pytesseract.tesseract_cmd set?")
                return None # Abort if Tesseract isn't found
            except Exception as te:
                print(f"Error during Tesseract OCR on page {page_num + 1}: {te}")
                extracted_pages_text.append(f"[OCR Error on page {page_num + 1}]")
    except Exception as e:
        print(f"Error processing PDF for Tesseract (PyMuPDF stage): {e}")
        return None
    finally:
        if doc:
            doc.close()
        if os.path.exists(TEMP_PAGE_IMAGE): # Though TEMP_PAGE_IMAGE isn't written to disk in this version
            os.remove(TEMP_PAGE_IMAGE)
    return extracted_pages_text


# --- Utility Functions ---

def save_text_to_file(text_pages, output_filename="extracted_text.txt", method_name="Unknown"):
    full_output_path = os.path.join(SCRIPT_DIR, output_filename)
    try:
        with open(full_output_path, "w", encoding="utf-8") as f:
            f.write(f"--- Extracted using method: {method_name} ---\n\n")
            if text_pages is None:
                f.write("No text was extracted or an error occurred.\n")
                print(f"No text extracted. Saved basic error message to: {full_output_path}")
                return
            for i, page_text in enumerate(text_pages):
                f.write(f"--- PAGE {i+1} ---\n")
                f.write(page_text if page_text else "[No text extracted for this page]\n")
                f.write("\n\n")
        print(f"Extracted text saved to: {full_output_path}")
    except Exception as e:
        print(f"Error saving text to file: {e}")

def display_page_text_interactively(text_pages, method_name="Unknown"):
    if not text_pages:
        print(f"No text to display (method: {method_name}).")
        return
    current_page_index = 0
    total_pages = len(text_pages)
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"--- Method: {method_name} | Page {current_page_index + 1} of {total_pages} ---")
        print("-" * 70)
        print(text_pages[current_page_index])
        print("-" * 70)
        print("\nCommands: 'n' (next), 'p' (prev), 's' (save), 'q' (quit), page number")
        command = input("Enter command: ").strip().lower()
        if command == 'q': break
        elif command == 'n': current_page_index = min(current_page_index + 1, total_pages - 1)
        elif command == 'p': current_page_index = max(current_page_index - 1, 0)
        elif command == 's':
            output_fn_default = f"extracted_text_{method_name.lower().replace(' ', '_')}.txt"
            output_fn = input(f"Save as ({output_fn_default}): ") or output_fn_default
            save_text_to_file(text_pages, output_fn, method_name)
            input("Press Enter to continue...")
        else:
            try:
                page_num = int(command)
                if 1 <= page_num <= total_pages: current_page_index = page_num - 1
                else: print(f"Invalid page number (1-{total_pages}).")
            except ValueError: print("Invalid command.")
            if command not in ['n','p','q','s']: input("Press Enter to continue...")


def save_comparison_to_file(pdf_path, page_number, results_dict, output_filename):
    """Saves the comparison results for a specific page to a single .txt file."""
    full_output_path = os.path.join(SCRIPT_DIR, output_filename)
    try:
        with open(full_output_path, "w", encoding="utf-8") as f:
            f.write(f"--- PDF Text Extraction Comparison ---\n")
            f.write(f"Source PDF: {os.path.basename(pdf_path)}\n")
            f.write(f"Page Compared: {page_number}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            for method_name, text_content in results_dict.items():
                f.write(f"--- Method: {method_name} ---\n")
                f.write(text_content if text_content else "[No text available for this page/method]\n")
                f.write("\n" + "=" * 70 + "\n\n")
        print(f"Comparison results saved to: {full_output_path}")
    except Exception as e:
        print(f"Error saving comparison file: {e}")

def display_single_page_text(extracted_text_per_page, page_num_to_display, method_name):
    """
    Displays the text of a single specified page from the extraction results.
    """
    # This initial check is mostly redundant if called correctly from main,
    # as main checks `if extracted_text_per_page:` before calling.
    # However, it's good for robustness if called from elsewhere.
    if not extracted_text_per_page: # Covers None (error) or empty list (0 pages)
        print(f"No text available from {method_name} to display page {page_num_to_display}.")
        if isinstance(extracted_text_per_page, list) and not extracted_text_per_page: # Specifically an empty list
             print(f"({method_name} returned 0 pages of text.)")
        return

    num_pages_extracted = len(extracted_text_per_page)
    page_idx = page_num_to_display - 1 # 0-indexed

    # This message is distinct from the ">>> Extracting with..." message from extraction functions
    print(f"--- Text for Page {page_num_to_display} (from {method_name.upper()}) ---")

    if method_name == 'pdfminer':
        # pdfminer's high-level extract_text returns all text as a single string (in a list of one item)
        if page_num_to_display == 1:
            print(f"(Note: pdfminer.six high-level extract_text returns all pages combined)")
            # Ensure there is text and it's not an empty string
            actual_text = extracted_text_per_page[0] if extracted_text_per_page and extracted_text_per_page[0] else "[Empty result from pdfminer]"
            print(actual_text)
        else:
            print(f"[pdfminer.six high-level extract_text does not isolate page {page_num_to_display}. "
                  f"Full document text is in its 'page 1' output.]")
        print("------------------------------------")
        return

    if 0 <= page_idx < num_pages_extracted:
        page_text = extracted_text_per_page[page_idx]
        if page_text: # Text found and is not an empty string
            print(page_text)
        else: # Text is an empty string for this page
            print(f"[No text content found on page {page_num_to_display} by {method_name}]")
    else: # Page number out of range for this method's results
        print(f"[Page {page_num_to_display} is out of range for {method_name}. "
              f"This method found {num_pages_extracted} page(s).]")
    print("------------------------------------")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Test various PDF to text conversion methods.")
    parser.add_argument(
        "pdf_file",
        nargs="?",
        default=os.path.join(SCRIPT_DIR, DEFAULT_PDF_FILENAME),
        help=f"Path to the PDF file (default: {DEFAULT_PDF_FILENAME} in script dir)."
    )
    parser.add_argument(
        "-m", "--method",
        choices=['pypdfloader', 'pymupdf', 'pdfminer', 'ocrmypdf', 'pytesseract', 'all'],
        default='pymupdf',
        help="Extraction method to use."
    )
    parser.add_argument(
        "-o", "--output",
        help="Save extracted text to the specified file (prefix). Appends method name."
    )
    parser.add_argument(
        "--compare-page",
        type=int,
        metavar="PAGE_NUM",
        help="Run all methods and output text of the specified page (1-indexed) to a single comparison file."
    )
    parser.add_argument(
        "--comparison-output",
        help="Filename for the consolidated comparison output. Used with --compare-page. Defaults to comparison_page_X_<pdf_basename>.txt"
    )
    parser.add_argument( # New argument
        "--page",
        type=int,
        metavar="PAGE_NUM",
        help="Display text of the specified page (1-indexed) for the selected method(s). "
             "If -o is also used, all text is saved to file AND this specified page is printed to console. "
             "Ignored if --compare-page is used."
    )

    args = parser.parse_args()
    pdf_path_to_test = args.pdf_file

    if args.page is not None and args.page < 1:
        print("Error: --page must be a positive integer (1-indexed).")
        return

    if not os.path.exists(pdf_path_to_test):
        if args.pdf_file == os.path.join(SCRIPT_DIR, DEFAULT_PDF_FILENAME):
            print(f"Default PDF '{DEFAULT_PDF_FILENAME}' not found in script directory.")
            print("Please create it or provide a PDF path as an argument.")
        else:
            print(f"Error: PDF file not found at '{pdf_path_to_test}'")
        print("Example: python your_script_name.py your_document.pdf -m pymupdf")
        return

    print(f"--- PDF Text Extraction Test on: {os.path.basename(pdf_path_to_test)} ---")

    method_choices_map = {
        'pypdfloader': extract_with_pypdfloader,
        'pymupdf': extract_with_pymupdf,
        'pdfminer': extract_with_pdfminer,
        'ocrmypdf': extract_with_ocrmypdf,
        'pytesseract': extract_with_pytesseract,
    }

    if args.compare_page:
        if args.compare_page < 1:
            print("Error: --compare-page must be 1 or greater.")
            return

        page_to_compare = args.compare_page
        print(f"\n--- Comparison Mode: Extracting page {page_to_compare} from all methods ---")

        comparison_results = {}

        for method_name, extract_func in method_choices_map.items():
            # Each extract_func should print its own ">>> Processing/Extracting with METHOD..." message
            extracted_text_per_page = extract_func(pdf_path_to_test)
            page_text_for_comparison = f"[No text extracted or error occurred with {method_name}]" # Default

            if extracted_text_per_page is not None: # Not None means extraction attempted, result is a list (possibly empty)
                if not extracted_text_per_page and isinstance(extracted_text_per_page, list): # Empty list means 0 pages
                     page_text_for_comparison = f"[Method {method_name} returned 0 pages of text]"
                elif method_name == 'pdfminer':
                    if page_to_compare == 1:
                        page_text_for_comparison = extracted_text_per_page[0] if extracted_text_per_page[0] else "[Empty result from pdfminer]"
                        page_text_for_comparison += "\n(Note: pdfminer.six high-level extract_text returns all pages combined)"
                    else:
                        page_text_for_comparison = f"[pdfminer.six high-level extract_text does not isolate page {page_to_compare}. Full document text is in its 'page 1' output.]"
                else: # For other methods that return a list of strings per page
                    if 0 <= page_to_compare - 1 < len(extracted_text_per_page):
                        page_text_for_comparison = extracted_text_per_page[page_to_compare - 1]
                        if not page_text_for_comparison: # Handle empty string for a page
                             page_text_for_comparison = f"[No text content found on page {page_to_compare} by {method_name}]"
                    else:
                        page_text_for_comparison = f"[Page {page_to_compare} not found or out of range for {method_name} (PDF has {len(extracted_text_per_page)} page(s) according to this method)]"
            
            comparison_results[method_name] = page_text_for_comparison

        if args.comparison_output:
            comparison_filename = args.comparison_output
        else:
            pdf_basename = os.path.splitext(os.path.basename(pdf_path_to_test))[0]
            comparison_filename = f"comparison_page_{page_to_compare}_{pdf_basename}.txt"
        
        save_comparison_to_file(pdf_path_to_test, page_to_compare, comparison_results, comparison_filename)
        print(f"\n--- Comparison for page {page_to_compare} complete. ---")
        # Clean up temp OCR PDF if it exists after comparison mode
        if os.path.exists(TEMP_OCR_PDF):
            try:
                os.remove(TEMP_OCR_PDF)
                print(f"Temporary OCR'd PDF '{TEMP_OCR_PDF}' removed after comparison.")
            except Exception as e:
                print(f"Warning: Could not remove temporary OCR'd PDF '{TEMP_OCR_PDF}' after comparison: {e}")
        return # End execution after comparison mode

    # --- Single/all method execution (if not in comparison mode) ---
    methods_to_run = {}
    if args.method == 'all':
        methods_to_run = method_choices_map
    elif args.method in method_choices_map:
        methods_to_run[args.method] = method_choices_map[args.method]
    else:
        print(f"Unknown method: {args.method}")
        return

    for method_name, extract_func in methods_to_run.items():
        extracted_text_per_page = None # Reset for each method
        try:
            # extract_func itself will print its ">>> Extracting with METHOD <<<" message
            extracted_text_per_page = extract_func(pdf_path_to_test)

            if extracted_text_per_page is not None: # Successfully extracted (result is list, possibly empty)
                num_pages = len(extracted_text_per_page)
                
                # Construct a meaningful page_info string
                if method_name == 'pdfminer':
                    if num_pages == 1 and extracted_text_per_page[0]:
                        page_info = "text (all combined) from 1 logical page entry"
                    else: # Covers empty list or list with one empty string
                        page_info = "0 pages (empty result from pdfminer or no text)"
                else:
                    page_info = f"{num_pages} page(s)"
                
                print(f"Successfully extracted {page_info} using {method_name}.")

                if args.output:
                    output_filename = f"{args.output}_{method_name}.txt"
                    save_text_to_file(extracted_text_per_page, output_filename, method_name)
                    # If --page is also specified, display that single page to console
                    if args.page:
                        display_single_page_text(extracted_text_per_page, args.page, method_name)
                
                elif args.page: # No --output, but --page is specified
                    display_single_page_text(extracted_text_per_page, args.page, method_name)
                
                elif args.method != 'all': # No --output, no --page, and it's a single method run
                    display_page_text_interactively(extracted_text_per_page, method_name)
                
                # If args.method == 'all', no --output, no --page:
                # No specific page/interactive display. Just success message above. This is intended.

            else: # extract_func returned None, indicating an error during extraction
                print(f"No text extracted or an error occurred with method: {method_name}.")
                if args.output:
                    output_filename = f"{args.output}_{method_name}.txt"
                    save_text_to_file(None, output_filename, method_name) # Save error/empty file

        except Exception as e: 
            print(f"An unexpected error occurred while processing method {method_name}: {e}")
            # import traceback # Uncomment for debugging
            # traceback.print_exc() # Uncomment for debugging
            if args.output:
                output_filename = f"{args.output}_{method_name}.txt"
                save_text_to_file(None, output_filename, f"{method_name} (Outer Loop Error: {e})")

        # Snippet display logic (modified to not show if args.page is used, as full page is shown then)
        if args.method == 'all' and extracted_text_per_page and args.output and not args.page:
            print(f"  Snippet from {method_name} (Page 1 / or all for pdfminer):")
            snippet_text = extracted_text_per_page[0] if extracted_text_per_page else ""
            snippet = (snippet_text[:200].replace('\n', ' ') + '...') if snippet_text else "[No text on first page]"
            print(f"    {snippet}\n")

    print("\n--- Test Ended ---")
    # General cleanup for TEMP_OCR_PDF
    if os.path.exists(TEMP_OCR_PDF):
        if args.compare_page:
            pass # Already handled removal in the --compare-page block
        elif not args.output: 
            # Kept for inspection if not using -o and not in comparison mode
            print(f"Intermediate OCR'd PDF '{TEMP_OCR_PDF}' kept for inspection (not using -o or --compare-page).")
        else: # args.output is true, not compare_page
             print(f"Intermediate OCR'd PDF '{TEMP_OCR_PDF}' kept as -o was specified (may be source for an output file).")

    # TEMP_PAGE_IMAGE is usually an internal temp for pytesseract and should be cleaned by its function.
    if os.path.exists(TEMP_PAGE_IMAGE):
        print(f"Warning: Temporary page image '{TEMP_PAGE_IMAGE}' still exists. Should ideally be cleaned by its generating function.")
        # You might add a forced removal here if necessary, e.g.:
        # try:
        #     os.remove(TEMP_PAGE_IMAGE)
        # except Exception as e:
        #     print(f"Could not remove {TEMP_PAGE_IMAGE}: {e}")

if __name__ == "__main__":
    main()