from docling.document_converter import DocumentConverter
from docling.document_converter import InputFormat, PdfFormatOption
from pathlib import Path
import pandas as pd
import json
import sys
import logging
import time
import os
import shutil
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"
        self.images_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        
        # Initialize DocumentConverter with default options
        # Don't try to set any custom options
        self.doc_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF]
        )

    def extract_all(self):
        """Extract all content from the PDF in a structured format."""
        logger.info(f"Processing {self.pdf_path}...")
        
        try:
            # Convert document based on documentation
            start_time = time.time()
            conv_result = self.doc_converter.convert(self.pdf_path)
            end_time = time.time() - start_time
            logger.info(f"Document converted in {end_time:.2f} seconds.")
            
            # Access the DoclingDocument
            docling_doc = conv_result.document
            
            # Create base structure
            document_data = {
                "metadata": {
                    "filename": self.pdf_path.name,
                    "conversion_time_seconds": round(end_time, 2)
                },
                "content": {},
                "pages": []
            }
            
            # Extract document metadata and content
            try:
                # Extract document title
                if hasattr(docling_doc, "name"):
                    document_data["metadata"]["title"] = docling_doc.name
                
                # Extract page count
                if hasattr(docling_doc, "num_pages"):
                    document_data["metadata"]["pages"] = docling_doc.num_pages()
                elif hasattr(docling_doc, "pages"):
                    document_data["metadata"]["pages"] = len(docling_doc.pages)
                    
                # Export document text content
                if hasattr(docling_doc, "export_to_text"):
                    document_data["content"]["text"] = docling_doc.export_to_text()
                
                # Export document markdown
                if hasattr(docling_doc, "export_to_markdown"):
                    document_data["content"]["markdown"] = docling_doc.export_to_markdown()
                
                # Process pages and their content
                if hasattr(docling_doc, "pages"):
                    for page_idx, (page_num, page) in enumerate(docling_doc.pages.items(), 1):
                        logger.info(f"Processing page {page_idx}")
                        
                        # Initialize page data structure
                        page_data = {
                            "page_number": page_idx,
                            "tables": [],
                            "footnotes": []
                        }
                        
                        # Extract page text
                        if hasattr(page, "export_to_text"):
                            page_data["content"] = page.export_to_text()
                        
                        # Process page dimensions if available
                        if hasattr(page, "size"):
                            page_data["dimensions"] = {
                                "width": page.size.width if hasattr(page.size, "width") else None,
                                "height": page.size.height if hasattr(page.size, "height") else None
                            }
                        
                        document_data["pages"].append(page_data)
                
                # Process tables from document
                if hasattr(docling_doc, "tables") and docling_doc.tables:
                    for table_idx, table in enumerate(docling_doc.tables, 1):
                        try:
                            # Get table data as DataFrame
                            df = None
                            try:
                                if hasattr(table, "export_to_dataframe"):
                                    df = table.export_to_dataframe()
                                elif hasattr(table, "to_dataframe"):
                                    df = table.to_dataframe()
                            except Exception as table_err:
                                logger.warning(f"Error converting table to DataFrame: {table_err}")
                                
                                # Attempt to build DataFrame from cells
                                if hasattr(table, "cells") and table.cells:
                                    try:
                                        rows = {}
                                        for cell in table.cells:
                                            row_idx = getattr(cell, "row", 0)
                                            col_idx = getattr(cell, "col", 0)
                                            text = ""
                                            if hasattr(cell, "export_to_text"):
                                                text = cell.export_to_text()
                                            elif hasattr(cell, "text"):
                                                text = cell.text
                                                
                                            if row_idx not in rows:
                                                rows[row_idx] = {}
                                            rows[row_idx][col_idx] = text
                                        
                                        # Convert to DataFrame
                                        if rows:
                                            data = []
                                            for row_idx in sorted(rows.keys()):
                                                row_data = []
                                                for col_idx in sorted(rows[row_idx].keys()):
                                                    row_data.append(rows[row_idx][col_idx])
                                                data.append(row_data)
                                            df = pd.DataFrame(data)
                                            logger.info(f"Built DataFrame from table cells")
                                    except Exception as cell_err:
                                        logger.warning(f"Error building DataFrame from cells: {cell_err}")
                                
                            if df is not None and not df.empty:
                                # Find page number for the table
                                page_num = None
                                if hasattr(table, "page_no"):
                                    page_num = table.page_no
                                elif hasattr(table, "page_number"):
                                    page_num = table.page_number
                                
                                # Save table to CSV
                                csv_path = self.tables_dir / f"{self.pdf_path.stem}_table_{table_idx}.csv"
                                if page_num:
                                    csv_path = self.tables_dir / f"{self.pdf_path.stem}_page_{page_num}_table_{table_idx}.csv"
                                
                                df.to_csv(csv_path, index=False, encoding='utf-8')
                                logger.info(f"Saved table to {csv_path}")
                                
                                # Get table caption if available
                                caption = None
                                if hasattr(table, "caption") and table.caption:
                                    if hasattr(table.caption, "export_to_text"):
                                        caption = table.caption.export_to_text()
                                    elif hasattr(table.caption, "text"):
                                        caption = table.caption.text
                                
                                # Create table data structure
                                table_data = {
                                    "id": f"table_{table_idx}",
                                    "caption": caption,
                                    "csv_path": str(csv_path),
                                    "rows": len(df),
                                    "columns": len(df.columns),
                                    "page_number": page_num
                                }
                                
                                # Add to the corresponding page
                                if page_num and 1 <= page_num <= len(document_data["pages"]):
                                    document_data["pages"][page_num-1]["tables"].append(table_data)
                                # If we can't determine the page, add to document-level tables
                                else:
                                    if "tables" not in document_data:
                                        document_data["tables"] = []
                                    document_data["tables"].append(table_data)
                        except Exception as e:
                            logger.warning(f"Error processing table {table_idx}: {str(e)}")
                
                # Generate PDF images directly using PyMuPDF 
                try:
                    import fitz  # PyMuPDF
                    logger.info("Extracting images using PyMuPDF")
                    pdf_document = fitz.open(self.pdf_path)
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        # Render page to image
                        pix = page.get_pixmap(alpha=False)
                        image_path = self.images_dir / f"{self.pdf_path.stem}_page_{page_num+1}.png"
                        pix.save(str(image_path))
                        logger.info(f"Saved page image {page_num+1}: {image_path}")
                        
                        # Add image path to the corresponding page data
                        if page_num < len(document_data["pages"]):
                            document_data["pages"][page_num]["image_path"] = str(image_path)
                        
                        # Extract images from the page
                        image_list = page.get_images(full=True)
                        for img_idx, img_info in enumerate(image_list):
                            try:
                                xref = img_info[0]  # Image reference
                                base_image = pdf_document.extract_image(xref)
                                image_bytes = base_image["image"]
                                image_ext = base_image["ext"]
                                
                                # Save individual image
                                img_filename = f"{self.pdf_path.stem}_page_{page_num+1}_img_{img_idx+1}.{image_ext}"
                                img_path = self.images_dir / img_filename
                                with open(img_path, "wb") as img_file:
                                    img_file.write(image_bytes)
                                logger.info(f"Saved embedded image: {img_path}")
                                
                                # Add to document structure
                                if "images" not in document_data:
                                    document_data["images"] = []
                                document_data["images"].append({
                                    "id": f"img_{page_num+1}_{img_idx+1}",
                                    "path": str(img_path),
                                    "page_number": page_num+1,
                                    "format": image_ext
                                })
                            except Exception as img_err:
                                logger.warning(f"Error saving embedded image: {img_err}")
                except ImportError:
                    logger.warning("PyMuPDF not installed. Run 'pip install pymupdf' to enable image extraction.")
                except Exception as e:
                    logger.warning(f"Error extracting images with PyMuPDF: {e}")
            
            except Exception as e:
                logger.error(f"Error during content extraction: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def save_structured_data(self):
        """Save all extracted data in a structured format."""
        try:
            # Extract all data
            data = self.extract_all()
            if not data:
                logger.error("No data was extracted from the document")
                return
            
            # Save main JSON structure
            json_path = self.output_dir / f"{self.pdf_path.stem}_structure.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save raw text if available
            if "content" in data and "text" in data["content"]:
                text_path = self.output_dir / f"{self.pdf_path.stem}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(data["content"]["text"])
                    
            # Save markdown if available
            if "content" in data and "markdown" in data["content"]:
                md_path = self.output_dir / f"{self.pdf_path.stem}.md"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(data["content"]["markdown"])
            
            # Check if any images were extracted
            if not os.listdir(self.images_dir):
                logger.warning("No images were extracted from the document")
            else:
                logger.info(f"Extracted {len(os.listdir(self.images_dir))} images")
            
            # Check if any tables were extracted
            if not os.listdir(self.tables_dir):
                logger.warning("No tables were extracted from the document")
            else:
                logger.info(f"Extracted {len(os.listdir(self.tables_dir))} tables")
            
            logger.info(f"\nExtraction complete!")
            logger.info(f"- JSON structure: {json_path}")
            logger.info(f"- Text content: {self.output_dir / f'{self.pdf_path.stem}.txt'}")
            logger.info(f"- Markdown content: {self.output_dir / f'{self.pdf_path.stem}.md'}")
            logger.info(f"- Images directory: {self.images_dir}")
            logger.info(f"- Tables directory: {self.tables_dir}")
            
            return data
                
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = "documents/tds_abt_es_1.pdf"
        
    extractor = DocumentExtractor(pdf_file)
    extractor.save_structured_data() 