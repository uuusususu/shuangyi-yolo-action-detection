from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "YOLO动作检测系统_简易操作手册.docx"
PACKAGE_OUTPUT = ROOT / "dist" / "YOLOActionDetection" / OUTPUT.name
MAIN_SCREENSHOT = ROOT / "assets" / "screenshots" / "main-window.png"
CONFIG_SCREENSHOT = ROOT / "assets" / "screenshots" / "config-page.png"

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "0B2545"
BODY = "243142"
MUTED = "667085"
CALLOUT_FILL = "E8EEF5"
CALLOUT_BORDER = "2E74B5"


def set_run_font(
    run,
    *,
    size: float,
    color: str = BODY,
    bold: bool = False,
    italic: bool = False,
    latin_font: str = "Calibri",
    cjk_font: str = "Microsoft YaHei",
) -> None:
    run.font.name = latin_font
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    run.bold = bold
    run.italic = italic
    r_pr = run._element.get_or_add_rPr()
    fonts = r_pr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, fonts)
    fonts.set(qn("w:ascii"), latin_font)
    fonts.set(qn("w:hAnsi"), latin_font)
    fonts.set(qn("w:eastAsia"), cjk_font)


def configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(BODY)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for style_name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 14, 7),
        ("Heading 3", 12, DARK_BLUE, 10, 5),
    ):
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    caption = doc.styles["Caption"]
    caption.font.name = "Calibri"
    caption.font.size = Pt(9)
    caption.font.italic = False
    caption.font.color.rgb = RGBColor.from_string(MUTED)
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_decimal_numbering(doc: Document) -> int:
    numbering = doc.part.numbering_part.element
    abstract_ids = [
        int(node.get(qn("w:abstractNumId")))
        for node in numbering.findall(qn("w:abstractNum"))
    ]
    num_ids = [
        int(node.get(qn("w:numId")))
        for node in numbering.findall(qn("w:num"))
    ]
    abstract_id = max(abstract_ids, default=0) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)

    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    level.append(start)
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "decimal")
    level.append(num_fmt)
    level_text = OxmlElement("w:lvlText")
    level_text.set(qn("w:val"), "%1.")
    level.append(level_text)
    justification = OxmlElement("w:lvlJc")
    justification.set(qn("w:val"), "left")
    level.append(justification)
    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "540")
    tabs.append(tab)
    p_pr.append(tabs)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "540")
    ind.set(qn("w:hanging"), "270")
    p_pr.append(ind)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:after"), "80")
    spacing.set(qn("w:line"), "300")
    spacing.set(qn("w:lineRule"), "auto")
    p_pr.append(spacing)
    level.append(p_pr)
    abstract.append(level)
    numbering.append(abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), str(abstract_id))
    num.append(abstract_ref)
    numbering.append(num)
    return num_id


def add_numbered_step(doc: Document, num_id: int, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.25
    num_pr = paragraph._element.get_or_add_pPr().get_or_add_numPr()
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num_pr.append(ilvl)
    num_id_node = OxmlElement("w:numId")
    num_id_node.set(qn("w:val"), str(num_id))
    num_pr.append(num_id_node)
    set_run_font(paragraph.add_run(text), size=11)


def add_callout(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(10)
    paragraph.paragraph_format.left_indent = Inches(0.12)
    paragraph.paragraph_format.right_indent = Inches(0.12)
    p_pr = paragraph._element.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), CALLOUT_FILL)
    p_pr.append(shading)
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:space"), "6")
    left.set(qn("w:color"), CALLOUT_BORDER)
    borders.append(left)
    p_pr.append(borders)
    set_run_font(paragraph.add_run(text), size=11, color=INK, bold=True)


def add_picture(doc: Document, path: Path, width: float, alt_text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run()
    shape = run.add_picture(str(path), width=Inches(width))
    doc_pr = shape._inline.docPr
    doc_pr.set("title", alt_text)
    doc_pr.set("descr", alt_text)


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    value = OxmlElement("w:t")
    value.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, value, end])
    set_run_font(run, size=9, color=MUTED)


def configure_page(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    header.paragraph_format.space_after = Pt(0)
    set_run_font(
        header.add_run("双翼科技视觉行为引导系统  |  简易操作手册"),
        size=9,
        color=MUTED,
        bold=True,
    )

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.paragraph_format.space_before = Pt(0)
    set_run_font(footer.add_run("现场操作员版  |  第 "), size=9, color=MUTED)
    add_page_field(footer)
    set_run_font(footer.add_run(" 页"), size=9, color=MUTED)


def add_title_block(doc: Document) -> None:
    kicker = doc.add_paragraph()
    kicker.paragraph_format.space_before = Pt(4)
    kicker.paragraph_format.space_after = Pt(2)
    set_run_font(kicker.add_run("现场操作员版"), size=10, color=BLUE, bold=True)

    title = doc.add_paragraph()
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(5)
    set_run_font(title.add_run("YOLO 动作检测系统"), size=27, color=INK, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_before = Pt(0)
    subtitle.paragraph_format.space_after = Pt(14)
    set_run_font(
        subtitle.add_run("简易操作手册  |  版本日期：2026-07-30"),
        size=11,
        color=MUTED,
    )


def build_document() -> Document:
    doc = Document()
    configure_styles(doc)
    configure_page(doc)
    doc.core_properties.title = "YOLO 动作检测系统简易操作手册"
    doc.core_properties.subject = "现场操作员启动、检测、配置与关闭说明"
    doc.core_properties.author = "双翼科技"

    add_title_block(doc)
    add_callout(doc, "操作顺序：打开相机 → 开始检测 → 停止检测 → 关闭相机")

    doc.add_heading("一、启动程序", level=1)
    startup_num = add_decimal_numbering(doc)
    add_numbered_step(doc, startup_num, "连接工业相机，确认相机已经通电并连接电脑。")
    add_numbered_step(doc, startup_num, "将整个 YOLOActionDetection 文件夹复制到电脑，不要只复制 EXE 文件。")
    add_numbered_step(doc, startup_num, "双击 start.bat，等待主界面显示 READY。")

    add_picture(doc, MAIN_SCREENSHOT, 5.9, "YOLO 动作检测系统主界面")
    doc.add_paragraph("图 1  主检测界面", style="Caption")

    doc.add_page_break()
    doc.add_heading("二、开始与停止检测", level=1)
    detect_num = add_decimal_numbering(doc)
    add_numbered_step(doc, detect_num, "点击“打开相机”，确认左侧画面能够正常显示。")
    add_numbered_step(doc, detect_num, "点击“开始检测”，系统进入检测状态。")
    add_numbered_step(doc, detect_num, "按照右侧显示的步骤顺序完成操作。识别成功后步骤状态会自动更新。")
    add_numbered_step(doc, detect_num, "需要暂停时点击“停止检测”；结束使用时再关闭相机。")
    add_callout(doc, "检测过程中不要拔插相机，也不要删除或重命名 config、_internal 文件夹。")

    doc.add_heading("ROI 模式检测异常告警", level=2)
    add_numbered_step(doc, detect_num, "ROI 模式下，当某轮判定为 NG 时，系统会自动弹出异常告警窗口。")
    add_numbered_step(doc, detect_num, "告警窗口冻结真正触发本轮 NG 的同一张 YOLO 推理画面，并严格裁剪为当前识别 ROI，不显示 ROI 外画面。")
    add_numbered_step(doc, detect_num, "裁剪图只用绿色旋转框标出中心点位于识别 ROI 内的目标；框线越过 ROI 边界时会自然裁掉。")
    add_numbered_step(doc, detect_num, "弹窗内容区只显示带框画面，不显示类别名、置信度、跟踪 ID、ROI 框、数量摘要或操作提示。")
    add_numbered_step(doc, detect_num, "识别 ROI 裁剪图保持宽高比，并自动放大到告警窗口的可用区域；正常主页面仍显示整张相机画面。")
    add_numbered_step(doc, detect_num, "告警期间 FAIL 提示音会循环播放，检测不会停止；若连续多轮 NG，窗口只替换为最新一轮对应的同帧检测画面。")
    add_numbered_step(doc, detect_num, "处理完毕后，按下键盘回车键（主键盘或小键盘均可）关闭告警窗口并停止声音。")
    add_callout(doc, "告警窗口弹出后检测继续进行，回车键只关闭窗口和声音，不会影响检测结果或统计。")

    doc.add_heading("保存 ROI NG 带框图片", level=2)
    add_numbered_step(doc, detect_num, "如需保存图片，在配置页“操作反馈”中开启“失败证据保存”，再点击“验证并保存”。未开启时不会保存 NG 图片。")
    add_numbered_step(doc, detect_num, "每个完成的 ROI NG 轮次只保存一张 frame.jpg；图片与弹窗使用同一推理帧和同一识别 ROI 裁剪，只显示 ROI 内目标的绿色旋转框。")
    add_numbered_step(doc, detect_num, "保存图片不显示类别名、置信度、跟踪 ID、ROI 或其他文字，也不会另外保存第二张无框原图。")
    add_numbered_step(doc, detect_num, "如果识别 ROI 内没有检测到任何目标，系统仍保存这张同帧无框裁剪图，不会生成虚拟框。")
    add_numbered_step(doc, detect_num, "metadata.json 只记录识别 ROI 内目标；目标坐标从裁剪图左上角开始，并记录裁剪区域在原图中的位置。")
    add_callout(doc, "保存图片位于 outputs/evidence 的日期与轮次目录中；图片保存失败不会影响检测、统计或报警。")

    doc.add_heading("三、选择相机与调整画面亮度", level=1)
    add_picture(doc, CONFIG_SCREENSHOT, 5.1, "YOLO 动作检测系统配置界面")
    doc.add_paragraph("图 2  配置界面", style="Caption")
    camera_num = add_decimal_numbering(doc)
    add_numbered_step(doc, camera_num, "点击主界面右上角“配置”，再选择左侧“工业相机”。")
    add_numbered_step(doc, camera_num, "点击“刷新设备”，在“迈德相机”下拉框中按设备 SN 选择现场相机。")
    add_numbered_step(doc, camera_num, "需要调整画面亮度时，将“参数模式”选择为 manual。")
    add_numbered_step(doc, camera_num, "调整“手动曝光（us）”：数值增大画面变亮，数值减小画面变暗。建议小幅调整后观察画面。")
    add_numbered_step(doc, camera_num, "点击“验证并保存”，再点击“返回检测”。相机选择和曝光参数保存后才会应用。")

    doc.add_page_break()
    doc.add_heading("四、选择模型", level=1)
    model_num = add_decimal_numbering(doc)
    add_numbered_step(doc, model_num, "点击“配置”，进入左侧“模型与步骤”。")
    add_numbered_step(doc, model_num, "在“模型文件路径”右侧点击“浏览”，选择需要使用的 .onnx 或 .pt 模型。")
    add_numbered_step(doc, model_num, "确认路径正确后点击“验证并保存”，程序会重新加载所选模型。")
    add_callout(doc, "模型文件保存后不要移动、改名或删除；配置类别必须与模型中的类别名称一致。")

    doc.add_heading("五、配置类别与数量", level=1)
    category_num = add_decimal_numbering(doc)
    add_numbered_step(doc, category_num, "在“模型与步骤”页面向下找到“类别步骤与数量”。")
    add_numbered_step(doc, category_num, "填写需要检查的类别名称；名称必须与当前模型类别完全一致。ROI 模式下各类别平级，不看填写顺序。")
    add_numbered_step(doc, category_num, "在“同帧目标数量”中填写该类别必须识别到的精确数量；少一个或多一个都不匹配。")
    add_numbered_step(doc, category_num, "不使用的后续步骤保持空白，全部设置完成后点击“验证并保存”。")
    add_callout(doc, "待检类别必须与当前模型类别完全一致，名称填写错误会导致检测无法开始。")

    doc.add_page_break()
    doc.add_heading("六、配置 ROI 定位检测", level=1)
    roi_num = add_decimal_numbering(doc)
    add_numbered_step(doc, roi_num, "先打开相机，再进入“配置”并选择左侧“ROI 配置”。")
    add_numbered_step(doc, roi_num, "点击“刷新画面”，使用相机最新静态画面配置区域。")
    add_numbered_step(doc, roi_num, "在“ROI 类型”下拉框选择“识别 ROI”，点击“创建/替换 ROI”，框住需要统计全部待检类别的固定区域，再点击编辑器下方“保存当前 ROI”。")
    add_numbered_step(doc, roi_num, "选择“模板 ROI”，点击“创建/替换 ROI”并框住一个轮廓清晰的定位目标，然后点击“保存当前 ROI”。保存模板 ROI 不要求搜索 ROI 或识别 ROI 已经配置。")
    add_numbered_step(doc, roi_num, "模板保存成功后，确认静态原图直接显示绿色真实轮廓、绿色外框和“创建得分”；显示红色失败原因时，旧模板和旧配置仍继续有效，可重新框选后再次保存。")
    add_numbered_step(doc, roi_num, "选择“搜索 ROI”，点击“创建/替换 ROI”并框住运行时允许查找模板的范围，再点击“保存当前 ROI”。三种 ROI 完全独立，不要求重叠或相互包含。")
    add_numbered_step(doc, roi_num, "设置模板匹配参数：“匹配分数阈值”（0-1，越高越严格，默认 0.6）、“旋转角度范围”（允许的旋转±度数，默认 15）、“角度步进”（默认 1）。")
    add_numbered_step(doc, roi_num, "三种 ROI 分别保存完成后，开启“ROI 定位检测”和“模板匹配定位”，再点击页面底部“验证并保存”统一应用运行配置。")
    add_callout(doc, "模板定位启用时，每帧只在搜索 ROI 内做固定比例边缘匹配；命中后才对整图运行 YOLO，再用固定识别 ROI 判断类别和数量。主画面只显示搜索 ROI 框。模板缺失、有效边缘不足或搜索 ROI 缺失时会阻止检测。相机位置、镜头、硬件裁剪或视野变化后，必须刷新画面并重新框选三种 ROI、创建模板。")

    doc.add_heading("七、设置检测节奏", level=1)
    rhythm_num = add_decimal_numbering(doc)
    add_numbered_step(doc, rhythm_num, "进入“动作判定”，设置“通过稳定帧数”。某类连续达到正确数量后会在本轮锁存 PASS。")
    add_numbered_step(doc, rhythm_num, "设置“NG 稳定帧数”。任意未完成类别连续数量不符达到该帧数，整轮判定 NG。")
    add_numbered_step(doc, rhythm_num, "设置“完成一轮间隔（秒）”。PASS 或 NG 后按该时间保持结果，间隔到期后允许下一轮。")
    add_numbered_step(doc, rhythm_num, "输送速度变化后，只需要重新调整完成一轮间隔，并现场连续检查节拍。")
    add_callout(doc, "每次 PASS 或 NG 只统计一轮；完成一轮间隔结束后才开始下一轮。")

    doc.add_heading("八、正常关闭", level=1)
    close_num = add_decimal_numbering(doc)
    add_numbered_step(doc, close_num, "先点击“停止检测”。")
    add_numbered_step(doc, close_num, "再点击“关闭相机”。")
    add_numbered_step(doc, close_num, "最后点击右上角“关闭”退出程序。")

    all_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    for required in (
        "刷新设备",
        "手动曝光（us）",
        "模型文件路径",
        "类别步骤与数量",
        "同帧目标数量",
        "保存当前 ROI",
        "匹配分数阈值",
        "旋转角度范围",
        "角度步进",
        "ROI 类型",
        "识别 ROI",
        "模板 ROI",
        "搜索 ROI",
        "创建/替换 ROI",
        "创建得分",
        "通过稳定帧数",
        "NG 稳定帧数",
        "完成一轮间隔（秒）",
        "只统计一轮",
        "异常告警",
        "绿色旋转框",
        "识别 ROI 裁剪图",
        "失败证据保存",
        "frame.jpg",
        "outputs/evidence",
        "回车键",
    ):
        if required not in all_text:
            raise ValueError(f"文档缺少用户要求内容：{required}")
    for forbidden in (
        "日志",
        "diagnose.bat",
        "startup_diagnostics",
        "portable_smoke",
        "搜索 ROI 必须完整包含模板 ROI",
        "已保存模板预览",
    ):
        if forbidden in all_text:
            raise ValueError(f"文档包含已取消内容：{forbidden}")
    return doc


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = build_document()
    document.save(OUTPUT)
    if PACKAGE_OUTPUT.parent.is_dir():
        shutil.copyfile(OUTPUT, PACKAGE_OUTPUT)
    print(OUTPUT)
    if PACKAGE_OUTPUT.exists():
        print(PACKAGE_OUTPUT)


if __name__ == "__main__":
    main()
