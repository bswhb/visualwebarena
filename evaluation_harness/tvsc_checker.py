"""
Triple-View Semantic Consistency (TVSC) Checker

This module implements the core TVSC framework for detecting adversarial UI perturbations
in multimodal web agents. It enforces consistency across three views:
1. Visual View - Screenshot/pixel-level rendering
2. Structural View - Accessibility Tree (A11y)
3. OCR View - Optical Character Recognition results

Reference: Cross-View Consistency Checking for Multimodal Web Agents under Adversarial UI Perturbations
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("[TVSC Warning] PaddleOCR not installed. OCR features will be limited.")

try:
    from playwright.sync_api import Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BoundingBox:
    """Represents a bounding box with (x1, y1, x2, y2) coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> 'BoundingBox':
        """Create from x, y, width, height format."""
        return cls(x, y, x + w, y + h)


@dataclass
class OCRResult:
    """Represents an OCR detection result."""
    text: str
    bbox: BoundingBox
    confidence: float
    polygon: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class A11yNode:
    """Represents a node in the Accessibility Tree."""
    node_id: str
    role: str
    name: str
    description: str
    bbox: Optional[BoundingBox]
    properties: Dict[str, Any] = field(default_factory=dict)
    children: List['A11yNode'] = field(default_factory=list)
    
    def get_text(self) -> str:
        """Get the combined text representation of this node."""
        text_parts = []
        if self.name:
            text_parts.append(self.name)
        if self.description:
            text_parts.append(self.description)
        if 'value' in self.properties:
            text_parts.append(str(self.properties['value']))
        return ' '.join(text_parts).strip()


@dataclass
class TripleViewElement:
    """An element represented across all three views."""
    element_id: int
    
    # Visual View
    visual_image: Optional[Image.Image] = None
    visual_bbox: Optional[BoundingBox] = None
    
    # Structural View
    structural_node: Optional[A11yNode] = None
    structural_text: str = ""
    structural_role: str = ""
    
    # OCR View
    ocr_results: List[OCRResult] = field(default_factory=list)
    ocr_text: str = ""
    
    # Consistency scores
    consistency_score: float = 0.0
    position_consistency: float = 0.0
    text_consistency: float = 0.0
    is_suspicious: bool = False


class ConsistencyDecision(Enum):
    """Decision result from consistency checking."""
    EXECUTE = "execute"      # Safe to execute
    REVIEW = "review"        # Needs cascade review
    REJECT = "reject"        # Block the action


# ============================================================================
# View Extractors
# ============================================================================

class VisualViewExtractor:
    """Extracts the Visual View from a web page."""
    
    def __init__(self, page: 'Page'):
        self.page = page
        self._screenshot_cache: Optional[Image.Image] = None
    
    def get_screenshot(self, force_refresh: bool = False) -> Image.Image:
        """Capture full-page screenshot as PIL Image."""
        if self._screenshot_cache is None or force_refresh:
            import io
            screenshot_bytes = self.page.screenshot(full_page=True)
            self._screenshot_cache = Image.open(io.BytesIO(screenshot_bytes))
        return self._screenshot_cache
    
    def get_element_visual(self, selector: str) -> Optional[Dict]:
        """Extract visual information for a specific element."""
        try:
            element = self.page.locator(selector)
            bbox_dict = element.bounding_box()
            if bbox_dict is None:
                return None
            
            import io
            element_screenshot = element.screenshot()
            element_image = Image.open(io.BytesIO(element_screenshot))
            
            bbox = BoundingBox.from_xywh(
                int(bbox_dict['x']), 
                int(bbox_dict['y']),
                int(bbox_dict['width']), 
                int(bbox_dict['height'])
            )
            
            return {
                'image': element_image,
                'bbox': bbox
            }
        except Exception as e:
            print(f"[VisualViewExtractor] Error extracting element: {e}")
            return None
    
    def crop_region(self, bbox: BoundingBox) -> Optional[Image.Image]:
        """Crop a region from the cached screenshot."""
        try:
            screenshot = self.get_screenshot()
            return screenshot.crop(bbox.to_tuple())
        except Exception:
            return None
    
    def clear_cache(self):
        """Clear the screenshot cache."""
        self._screenshot_cache = None


class StructuralViewExtractor:
    """Extracts the Structural View from the Accessibility Tree."""
    
    # Interactive roles that we care about
    INTERACTIVE_ROLES = {
        'button', 'link', 'textbox', 'checkbox', 'radio',
        'combobox', 'menuitem', 'tab', 'switch', 'searchbox',
        'option', 'menuitemcheckbox', 'menuitemradio', 'slider',
        'spinbutton', 'treeitem', 'gridcell', 'row', 'cell'
    }
    
    def __init__(self, page: 'Page'):
        self.page = page
        self._cdp_session = None
    
    @property
    def cdp_session(self):
        """Lazy initialization of CDP session."""
        if self._cdp_session is None:
            self._cdp_session = self.page.context.new_cdp_session(self.page)
            self._cdp_session.send('Accessibility.enable')
        return self._cdp_session
    
    def get_accessibility_tree(self) -> Dict:
        """Retrieve the full accessibility tree via CDP."""
        try:
            tree = self.cdp_session.send('Accessibility.getFullAXTree')
            return tree
        except Exception as e:
            print(f"[StructuralViewExtractor] Error getting a11y tree: {e}")
            return {'nodes': []}
    
    def parse_node(self, node: Dict) -> A11yNode:
        """Parse a raw accessibility node into structured format."""
        properties = {}
        for prop in node.get('properties', []):
            prop_name = prop.get('name', '')
            prop_value = prop.get('value', {}).get('value')
            if prop_name and prop_value is not None:
                properties[prop_name] = prop_value
        
        # Extract bounding box if available
        bbox = None
        if 'boundingBox' in node:
            bb = node['boundingBox']
            bbox = BoundingBox.from_xywh(
                int(bb.get('x', 0)), 
                int(bb.get('y', 0)),
                int(bb.get('width', 0)), 
                int(bb.get('height', 0))
            )
        
        return A11yNode(
            node_id=node.get('nodeId', ''),
            role=node.get('role', {}).get('value', 'unknown'),
            name=node.get('name', {}).get('value', ''),
            description=node.get('description', {}).get('value', ''),
            bbox=bbox,
            properties=properties,
            children=[]
        )
    
    def get_interactive_elements(self) -> List[A11yNode]:
        """Extract all interactive elements from accessibility tree."""
        tree = self.get_accessibility_tree()
        interactive_elements = []
        
        def traverse(nodes: List[Dict]):
            for node in nodes:
                parsed = self.parse_node(node)
                if parsed.role.lower() in self.INTERACTIVE_ROLES:
                    interactive_elements.append(parsed)
                if 'children' in node:
                    traverse(node['children'])
        
        if 'nodes' in tree:
            traverse(tree['nodes'])
        
        return interactive_elements
    
    def get_element_by_selector(self, selector: str) -> Optional[A11yNode]:
        """Get accessibility info for a specific element by selector."""
        try:
            # Use JavaScript to get the element's accessibility info
            a11y_info = self.page.evaluate(f'''
                () => {{
                    const el = document.querySelector("{selector}");
                    if (!el) return null;
                    const rect = el.getBoundingClientRect();
                    return {{
                        role: el.getAttribute('role') || el.tagName.toLowerCase(),
                        name: el.getAttribute('aria-label') || el.innerText || el.textContent || '',
                        description: el.getAttribute('aria-description') || '',
                        bbox: {{
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        }}
                    }};
                }}
            ''')
            
            if a11y_info is None:
                return None
            
            bbox = BoundingBox.from_xywh(
                int(a11y_info['bbox']['x']),
                int(a11y_info['bbox']['y']),
                int(a11y_info['bbox']['width']),
                int(a11y_info['bbox']['height'])
            )
            
            return A11yNode(
                node_id=selector,
                role=a11y_info['role'],
                name=a11y_info['name'][:200],  # Truncate long text
                description=a11y_info['description'],
                bbox=bbox
            )
        except Exception as e:
            print(f"[StructuralViewExtractor] Error getting element: {e}")
            return None


class OCRViewExtractor:
    """Extracts the OCR View from page screenshots."""
    
    def __init__(self, use_gpu: bool = False, lang: str = 'en'):
        self.ocr = None
        self.use_gpu = use_gpu
        self.lang = lang
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of OCR model."""
        if not self._initialized and PADDLEOCR_AVAILABLE:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False
            )
            self._initialized = True
    
    def extract_text_regions(self, image: Image.Image) -> List[OCRResult]:
        """Extract all text regions from an image."""
        if not PADDLEOCR_AVAILABLE:
            return []
        
        self._ensure_initialized()
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Run OCR
        try:
            results = self.ocr.ocr(img_array, cls=True)
        except Exception as e:
            print(f"[OCRViewExtractor] OCR error: {e}")
            return []
        
        ocr_results = []
        if results and results[0]:
            for line in results[0]:
                polygon = line[0]  # 4 corner points
                text = line[1][0]  # Recognized text
                confidence = line[1][1]  # Confidence score
                
                # Convert polygon to bounding box
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                bbox = BoundingBox(
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                )
                
                ocr_results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    polygon=[(int(p[0]), int(p[1])) for p in polygon]
                ))
        
        return ocr_results
    
    def extract_text_in_region(self, image: Image.Image, 
                                bbox: BoundingBox) -> List[OCRResult]:
        """Extract text only within a specified bounding box region."""
        try:
            cropped = image.crop(bbox.to_tuple())
            results = self.extract_text_regions(cropped)
            
            # Adjust coordinates back to original image space
            for r in results:
                r.bbox = BoundingBox(
                    r.bbox.x1 + bbox.x1,
                    r.bbox.y1 + bbox.y1,
                    r.bbox.x2 + bbox.x1,
                    r.bbox.y2 + bbox.y1
                )
            
            return results
        except Exception:
            return []


# ============================================================================
# Consistency Checker
# ============================================================================

class TripleViewConsistencyChecker:
    """
    Main TVSC checker that computes consistency scores across three views
    and makes decisions about action safety.
    """
    
    # Thresholds for decision making
    DEFAULT_EXECUTE_THRESHOLD = 0.7
    DEFAULT_REVIEW_THRESHOLD = 0.4
    
    # Weights for consistency components
    DEFAULT_WEIGHTS = {
        'position': 0.4,
        'text': 0.5,
        'style': 0.1
    }
    
    # Patterns for detecting suspicious content
    INJECTION_PATTERNS = [
        r'ignore\s+(previous|all|above)\s+instructions?',
        r'your\s+new\s+(task|instruction|goal)\s+is',
        r'disregard\s+(everything|all)',
        r'forget\s+(everything|all|previous)',
        r'override\s+(instructions?|commands?)',
        r'system\s*:\s*',
        r'\[INST\]',
        r'<\|im_start\|>',
    ]
    
    SUSPICIOUS_UI_PATTERNS = [
        r'verify\s+your\s+account\s+immediately',
        r'your\s+account\s+(has\s+been|is)\s+(compromised|hacked)',
        r'click\s+here\s+to\s+(claim|verify|confirm)',
        r'urgent\s*[:\!]',
        r'security\s+alert\s*[:\!]',
        r'password\s+expired',
        r'session\s+expired.*click',
    ]
    
    def __init__(
        self,
        page: 'Page',
        execute_threshold: float = None,
        review_threshold: float = None,
        weights: Dict[str, float] = None,
        use_gpu: bool = False
    ):
        self.page = page
        
        # Initialize extractors
        self.visual_extractor = VisualViewExtractor(page)
        self.structural_extractor = StructuralViewExtractor(page)
        self.ocr_extractor = OCRViewExtractor(use_gpu=use_gpu)
        
        # Set thresholds
        self.execute_threshold = execute_threshold or self.DEFAULT_EXECUTE_THRESHOLD
        self.review_threshold = review_threshold or self.DEFAULT_REVIEW_THRESHOLD
        
        # Set weights
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Compile regex patterns
        self._injection_re = re.compile(
            '|'.join(self.INJECTION_PATTERNS), 
            re.IGNORECASE
        )
        self._suspicious_ui_re = re.compile(
            '|'.join(self.SUSPICIOUS_UI_PATTERNS),
            re.IGNORECASE
        )
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'execute_count': 0,
            'review_count': 0,
            'reject_count': 0,
            'injection_detected': 0,
            'suspicious_ui_detected': 0
        }
    
    def compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.area
        area2 = box2.area
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using Levenshtein ratio."""
        if not text1 or not text2:
            return 0.0 if (text1 or text2) else 1.0
        
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        if t1 == t2:
            return 1.0
        
        # Simple Levenshtein ratio
        len1, len2 = len(t1), len(t2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Create distance matrix
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if t1[i-1] == t2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        distance = dp[len1][len2]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)
    
    def compute_position_consistency(
        self, 
        visual_bbox: Optional[BoundingBox],
        structural_bbox: Optional[BoundingBox],
        ocr_bbox: Optional[BoundingBox]
    ) -> float:
        """Compute position consistency across views."""
        scores = []
        
        if visual_bbox and structural_bbox:
            scores.append(self.compute_iou(visual_bbox, structural_bbox))
        
        if visual_bbox and ocr_bbox:
            scores.append(self.compute_iou(visual_bbox, ocr_bbox))
        
        if structural_bbox and ocr_bbox:
            scores.append(self.compute_iou(structural_bbox, ocr_bbox))
        
        return np.mean(scores) if scores else 0.0
    
    def compute_text_consistency(
        self,
        structural_text: str,
        ocr_text: str
    ) -> float:
        """Compute text consistency between structural and OCR views."""
        return self.compute_text_similarity(structural_text, ocr_text)
    
    def compute_consistency_score(self, element: TripleViewElement) -> float:
        """Compute the overall consistency score for an element."""
        # Position consistency
        ocr_bbox = element.ocr_results[0].bbox if element.ocr_results else None
        c_p = self.compute_position_consistency(
            element.visual_bbox,
            element.structural_node.bbox if element.structural_node else None,
            ocr_bbox
        )
        element.position_consistency = c_p
        
        # Text consistency
        c_t = self.compute_text_consistency(
            element.structural_text,
            element.ocr_text
        )
        element.text_consistency = c_t
        
        # Weighted combination
        score = (
            self.weights['position'] * c_p +
            self.weights['text'] * c_t
            # Style consistency can be added here
        )
        
        element.consistency_score = score
        return score
    
    def detect_injection(self, text: str) -> bool:
        """Detect prompt injection patterns in text."""
        if self._injection_re.search(text):
            self.stats['injection_detected'] += 1
            return True
        return False
    
    def detect_suspicious_ui(self, text: str) -> bool:
        """Detect suspicious UI patterns (phishing, etc.)."""
        if self._suspicious_ui_re.search(text):
            self.stats['suspicious_ui_detected'] += 1
            return True
        return False
    
    def check_element(
        self,
        selector: str,
        is_critical_action: bool = False
    ) -> Tuple[ConsistencyDecision, TripleViewElement]:
        """
        Check consistency for a single element and make a decision.
        
        Args:
            selector: CSS selector for the element
            is_critical_action: Whether this is a critical action (higher threshold)
        
        Returns:
            Tuple of (decision, element_info)
        """
        self.stats['total_checks'] += 1
        
        # Extract Visual View
        screenshot = self.visual_extractor.get_screenshot()
        visual_info = self.visual_extractor.get_element_visual(selector)
        visual_bbox = visual_info['bbox'] if visual_info else None
        visual_image = visual_info['image'] if visual_info else None
        
        # Extract Structural View
        structural_node = self.structural_extractor.get_element_by_selector(selector)
        structural_text = structural_node.get_text() if structural_node else ""
        structural_role = structural_node.role if structural_node else ""
        
        # Extract OCR View
        ocr_results = []
        ocr_text = ""
        if visual_bbox:
            # Expand bbox slightly for OCR
            expanded_bbox = BoundingBox(
                max(0, visual_bbox.x1 - 5),
                max(0, visual_bbox.y1 - 5),
                visual_bbox.x2 + 5,
                visual_bbox.y2 + 5
            )
            ocr_results = self.ocr_extractor.extract_text_in_region(
                screenshot, expanded_bbox
            )
            ocr_text = ' '.join([r.text for r in ocr_results])
        
        # Create TripleViewElement
        element = TripleViewElement(
            element_id=hash(selector),
            visual_image=visual_image,
            visual_bbox=visual_bbox,
            structural_node=structural_node,
            structural_text=structural_text,
            structural_role=structural_role,
            ocr_results=ocr_results,
            ocr_text=ocr_text
        )
        
        # Check for injection/suspicious patterns
        all_text = f"{structural_text} {ocr_text}"
        if self.detect_injection(all_text) or self.detect_suspicious_ui(all_text):
            element.is_suspicious = True
            self.stats['reject_count'] += 1
            return ConsistencyDecision.REJECT, element
        
        # Compute consistency score
        score = self.compute_consistency_score(element)
        
        # Apply higher threshold for critical actions
        execute_thresh = self.execute_threshold
        review_thresh = self.review_threshold
        if is_critical_action:
            execute_thresh = min(0.85, execute_thresh + 0.15)
            review_thresh = min(0.6, review_thresh + 0.15)
        
        # Make decision
        if score >= execute_thresh:
            self.stats['execute_count'] += 1
            return ConsistencyDecision.EXECUTE, element
        elif score >= review_thresh:
            self.stats['review_count'] += 1
            return ConsistencyDecision.REVIEW, element
        else:
            element.is_suspicious = True
            self.stats['reject_count'] += 1
            return ConsistencyDecision.REJECT, element
    
    def check_all_interactive_elements(self) -> List[TripleViewElement]:
        """
        Extract and check all interactive elements on the page.
        
        Returns:
            List of TripleViewElement with consistency scores
        """
        screenshot = self.visual_extractor.get_screenshot()
        full_ocr = self.ocr_extractor.extract_text_regions(screenshot)
        structural_elements = self.structural_extractor.get_interactive_elements()
        
        results = []
        
        for idx, struct_elem in enumerate(structural_elements):
            if struct_elem.bbox is None:
                continue
            
            # Extract visual patch
            try:
                visual_image = screenshot.crop(struct_elem.bbox.to_tuple())
            except Exception:
                visual_image = None
            
            # Find overlapping OCR results
            overlapping_ocr = [
                ocr for ocr in full_ocr
                if self.compute_iou(ocr.bbox, struct_elem.bbox) > 0.3
            ]
            ocr_text = ' '.join([r.text for r in overlapping_ocr])
            
            element = TripleViewElement(
                element_id=idx,
                visual_image=visual_image,
                visual_bbox=struct_elem.bbox,
                structural_node=struct_elem,
                structural_text=struct_elem.get_text(),
                structural_role=struct_elem.role,
                ocr_results=overlapping_ocr,
                ocr_text=ocr_text
            )
            
            # Compute consistency
            self.compute_consistency_score(element)
            
            # Check for suspicious content
            all_text = f"{element.structural_text} {element.ocr_text}"
            if self.detect_injection(all_text) or self.detect_suspicious_ui(all_text):
                element.is_suspicious = True
            
            results.append(element)
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get checker statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
        self.visual_extractor.clear_cache()


# ============================================================================
# Observation Filter (Heuristic-based pre-filter)
# ============================================================================

class ObservationFilter:
    """
    Lightweight heuristic filter for observations.
    Operates before the full TVSC check to catch obvious attacks.
    """
    
    INJECTION_PATTERNS = [
        (r'ignore\s+(previous|all|above)\s+instructions?', 'injection'),
        (r'your\s+new\s+(task|instruction|goal)\s+is', 'injection'),
        (r'disregard\s+(everything|all)', 'injection'),
        (r'forget\s+(everything|all|previous)', 'injection'),
        (r'system\s*:\s*', 'injection'),
    ]
    
    PHISHING_PATTERNS = [
        (r'verify\s+your\s+account\s+immediately', 'phishing'),
        (r'account\s+(compromised|hacked|suspended)', 'phishing'),
        (r'click\s+here\s+to\s+(claim|verify|confirm)', 'phishing'),
        (r'urgent\s*[:\!].*password', 'phishing'),
    ]
    
    OVERLAY_PATTERNS = [
        (r'position\s*:\s*fixed', 'overlay'),
        (r'z-index\s*:\s*\d{4,}', 'overlay'),
    ]
    
    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive
        self.all_patterns = (
            self.INJECTION_PATTERNS + 
            self.PHISHING_PATTERNS + 
            self.OVERLAY_PATTERNS
        )
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), t) for p, t in self.all_patterns
        ]
        
        self.stats = {
            'total_filtered': 0,
            'attack_types': {}
        }
    
    def filter_observation(self, observation: Dict) -> Dict:
        """
        Filter an observation dictionary and sanitize suspicious content.
        
        Args:
            observation: Raw observation from browser
        
        Returns:
            Filtered observation with 'filtered' flag and 'detected_attacks' list
        """
        text = observation.get('text', '')
        detected_attacks = []
        
        for pattern, attack_type in self._compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                detected_attacks.append({
                    'type': attack_type,
                    'matches': matches
                })
                self.stats['attack_types'][attack_type] = \
                    self.stats['attack_types'].get(attack_type, 0) + 1
        
        # Sanitize if attacks detected
        filtered_text = text
        if detected_attacks:
            self.stats['total_filtered'] += 1
            
            if self.aggressive:
                # Remove suspicious patterns entirely
                for pattern, _ in self._compiled_patterns:
                    filtered_text = pattern.sub('[FILTERED]', filtered_text)
            else:
                # Just mark them
                for pattern, _ in self._compiled_patterns:
                    filtered_text = pattern.sub(
                        r'[SUSPICIOUS: \g<0>]', filtered_text
                    )
        
        return {
            **observation,
            'text': filtered_text,
            'filtered': len(detected_attacks) > 0,
            'detected_attacks': detected_attacks
        }
    
    def get_stats(self) -> Dict:
        """Get filter statistics."""
        return self.stats.copy()


# ============================================================================
# Robust Agent Wrapper
# ============================================================================

class RobustAgentWrapper:
    """
    Wrapper that adds TVSC-based robustness to any base agent.
    """
    
    # Actions that require higher scrutiny
    CRITICAL_ACTIONS = {
        'buy', 'purchase', 'pay', 'checkout', 'confirm', 'delete',
        'remove', 'submit', 'send', 'transfer', 'withdraw'
    }
    
    def __init__(
        self,
        page: 'Page',
        use_gpu: bool = False,
        enable_filter: bool = True,
        enable_tvsc: bool = True
    ):
        self.page = page
        self.enable_filter = enable_filter
        self.enable_tvsc = enable_tvsc
        
        # Initialize components
        self.observation_filter = ObservationFilter(aggressive=False)
        self.tvsc_checker = TripleViewConsistencyChecker(page, use_gpu=use_gpu)
        
        # Attack log
        self.attack_log = []
    
    def is_critical_action(self, action_text: str) -> bool:
        """Check if the action is critical based on its text."""
        action_lower = action_text.lower()
        return any(keyword in action_lower for keyword in self.CRITICAL_ACTIONS)
    
    def preprocess_observation(self, observation: Dict) -> Dict:
        """Pre-process observation through the filter."""
        if not self.enable_filter:
            return observation
        
        filtered = self.observation_filter.filter_observation(observation)
        
        if filtered['detected_attacks']:
            self.attack_log.append({
                'type': 'filter',
                'attacks': filtered['detected_attacks']
            })
        
        return filtered
    
    def check_action_target(
        self,
        selector: str,
        action_text: str = ""
    ) -> Tuple[bool, str]:
        """
        Check if it's safe to perform an action on a target element.
        
        Returns:
            Tuple of (is_safe, reason)
        """
        if not self.enable_tvsc:
            return True, "TVSC disabled"
        
        is_critical = self.is_critical_action(action_text)
        decision, element = self.tvsc_checker.check_element(
            selector, is_critical_action=is_critical
        )
        
        if decision == ConsistencyDecision.EXECUTE:
            return True, f"Consistency score: {element.consistency_score:.2f}"
        
        elif decision == ConsistencyDecision.REVIEW:
            # Log and allow with warning
            self.attack_log.append({
                'type': 'review',
                'selector': selector,
                'score': element.consistency_score
            })
            return True, f"Review needed (score: {element.consistency_score:.2f})"
        
        else:  # REJECT
            self.attack_log.append({
                'type': 'reject',
                'selector': selector,
                'score': element.consistency_score,
                'is_suspicious': element.is_suspicious
            })
            return False, f"Rejected (score: {element.consistency_score:.2f}, suspicious: {element.is_suspicious})"
    
    def get_defense_stats(self) -> Dict:
        """Get comprehensive defense statistics."""
        return {
            'filter_stats': self.observation_filter.get_stats(),
            'tvsc_stats': self.tvsc_checker.get_stats(),
            'attack_log': self.attack_log
        }
    
    def reset(self):
        """Reset all statistics and caches."""
        self.attack_log = []
        self.tvsc_checker.reset_stats()
