# About and Support Features

This document describes the newly added About dialog and Support features in MinerU2PPTX.

## Features Overview

### About Dialog
- **Location**: Accessible via the "About" button in the top-right corner of the main interface
- **Content**:
  - Application name and version information
  - Description of the tool's functionality
  - Developer information section with GitHub profile
  - GitHub profile photo (fetched dynamically from GitHub API)
  - Clickable GitHub link to visit the developer's profile
  - Support section with Buy Me a Coffee integration
  - Translated content supporting English, Hungarian, and Chinese

### Post-Conversion Support Dialog
- **Trigger**: Automatically appears after successful conversion completion
- **Purpose**: Encourage users to support the developer
- **Content**:
  - Success confirmation message
  - Support message encouraging donations
  - Quick access buttons to Buy Me a Coffee and GitHub
  - Translated content in all supported languages

## Technical Implementation

### New Classes

#### `AboutDialog(tk.Toplevel)`
- Modal dialog window with developer information
- Dynamic GitHub photo loading with fallback to emoji
- Proper window positioning and sizing
- Multi-language support through translation system

#### `SupportDialog(tk.Toplevel)`
- Modal dialog shown after conversion success
- Automatic display with configurable delay
- Quick access to support links
- User-friendly success feedback

### Integration Points

#### Main GUI (`gui.py`)
- Added "About" button to `top_controls_frame`
- New methods: `_show_about_dialog()` and `_show_support_dialog()`
- Modified `_finalize_gui()` to show support dialog on success
- Updated `_update_ui_text()` for About button localization

#### Translation Updates
All translation files (`en.json`, `hu.json`, `zh.json`) updated with:
- `about_button`: "About" button text
- `about_title`: About dialog window title
- `about_app_name`: Application name in About dialog
- `about_version`: Version information
- `about_description`: Application description
- `about_developer`: Developer section label
- `about_github`: GitHub section label
- `about_support`: Support section label
- `support_button`: Buy Me a Coffee button text
- `github_button`: GitHub visit button text
- `support_message`: Post-conversion support message
- `support_dialog_title`: Support dialog window title

### External Dependencies

#### New Dependencies Added
- `urllib.request`: For downloading GitHub profile photos
- `PIL` (Pillow): For image processing and display (already in requirements.txt)

#### Links and Integrations
- **GitHub Profile**: https://github.com/arlinamid
- **Buy Me a Coffee**: https://buymeacoffee.com/arlinamid
- **GitHub Avatar API**: https://avatars.githubusercontent.com/arlinamid

## User Experience Flow

### About Dialog Access
1. User clicks "About" button in top-right corner
2. Modal About dialog opens with developer information
3. User can view GitHub profile, download photo, and access support links
4. Dialog closes when user clicks "Close" or navigates away

### Post-Conversion Support
1. User completes a successful conversion (single or batch mode)
2. Standard completion message appears first
3. After 500ms delay, Support dialog automatically appears
4. User can choose to support via Buy Me a Coffee or visit GitHub
5. Both actions close the dialog and open external links

## Localization

### Supported Languages
- **English**: Full support with natural phrasing
- **Hungarian**: Complete translation with culturally appropriate terms
- **Chinese**: Comprehensive translation with proper character encoding

### Language-Specific Considerations
- Hungarian: Uses formal address forms and proper technical terms
- Chinese: Simplified characters with standard technical vocabulary
- All languages: Maintain consistent tone and professional presentation

## Error Handling

### GitHub Photo Loading
- Primary: Dynamic download from GitHub API
- Fallback 1: Generic user emoji (👤) if download fails
- Fallback 2: Text-based placeholder if emoji fails
- Temporary file cleanup after successful photo loading

### Network Dependencies
- GitHub photo loading is non-blocking
- Graceful degradation for offline scenarios
- External link opening uses system default browser
- No network failures prevent dialog functionality

## Future Enhancements

### Potential Improvements
- Add application changelog to About dialog
- Include system information diagnostics
- Add social media links or project documentation
- Implement donation tracking or thank you messages
- Add keyboard shortcuts for About dialog access

### Maintenance Considerations
- Update version numbers in translation files
- Monitor GitHub API rate limits for photo loading
- Verify external link availability
- Test dialog behavior across different screen sizes
- Ensure proper cleanup of temporary image files